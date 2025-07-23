#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Advanced Options GUI - Schwabot Intelligent Compression & Storage Management
==============================================================================

Provides comprehensive advanced options for:
- Alpha Encryption-based intelligent compression
- Multi-device storage management
- Progressive learning configuration
- Educational content and setup guidance

Developed by Maxamillion M.A.A. DeLeon ("The Schwa") & Nexus AI
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path
from datetime import datetime

# Import compression manager
try:
    from alpha_compression_manager import (
        get_storage_device_manager, 
        StorageDevice, 
        AlphaCompressionManager,
        compress_trading_data_on_device,
        auto_compress_device_data,
        get_device_compression_suggestions
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

logger = None  # Will be set by the launcher


class AdvancedOptionsGUI:
    """
    🔧 Advanced Options GUI for Schwabot
    
    Provides comprehensive configuration options for intelligent compression
    and storage management across multiple devices.
    """
    
    def __init__(self, parent=None):
        self.parent = parent
        self.root = None
        self.device_manager = None
        self.selected_device = None
        self.device_listbox = None
        self.compression_status_label = None
        self.stats_text = None
        
        if COMPRESSION_AVAILABLE:
            self.device_manager = get_storage_device_manager()
    
    def show_advanced_options(self):
        """Show the main Advanced Options interface."""
        self.root = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.root.title("🔧 Schwabot Advanced Options")
        self.root.geometry("1000x800")
        self.root.configure(bg='#1e1e1e')
        
        # Configure style
        self._configure_styles()
        
        # Create main interface
        self._create_header()
        self._create_main_content()
        self._create_status_bar()
        
        # Initialize device detection
        if COMPRESSION_AVAILABLE:
            self._detect_devices()
        
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
                       foreground='#00ccff', 
                       background='#1e1e1e')
        
        style.configure('Info.TLabel', 
                       font=('Arial', 10), 
                       foreground='#cccccc', 
                       background='#1e1e1e')
        
        style.configure('Success.TLabel', 
                       font=('Arial', 10, 'bold'), 
                       foreground='#00ff00', 
                       background='#1e1e1e')
        
        style.configure('Warning.TLabel', 
                       font=('Arial', 10, 'bold'), 
                       foreground='#ffaa00', 
                       background='#1e1e1e')
        
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TNotebook', background='#1e1e1e')
        style.configure('TNotebook.Tab', background='#404040', foreground='white', padding=[10, 5])
    
    def _create_header(self):
        """Create the header section."""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=20, pady=10)
        
        # Title
        title_label = ttk.Label(header_frame, text="🔧 Advanced Options", style='Title.TLabel')
        title_label.pack()
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, 
                                  text="Intelligent Compression & Storage Management", 
                                  style='Info.TLabel')
        subtitle_label.pack(pady=5)
    
    def _create_main_content(self):
        """Create the main content area with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create tabs
        self._create_compression_tab()
        self._create_storage_tab()
        self._create_learning_tab()
        self._create_scheduling_tab()  # New scheduling tab
        self._create_education_tab()
        self._create_settings_tab()
    
    def _create_compression_tab(self):
        """Create the intelligent compression configuration tab."""
        compression_frame = ttk.Frame(self.notebook)
        self.notebook.add(compression_frame, text="🔐 Intelligent Compression")
        
        # Title
        title = ttk.Label(compression_frame, text="Alpha Encryption-Based Intelligent Compression", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(compression_frame, 
                        text="Leverage Alpha Encryption (Ω-B-Γ Logic) for intelligent data compression\n"
                             "that not only saves space but improves trading performance through progressive learning.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=5)
        
        # Device selection frame
        device_frame = ttk.LabelFrame(compression_frame, text="Storage Device Selection", padding=10)
        device_frame.pack(fill='x', padx=20, pady=10)
        
        # Device list
        device_list_frame = ttk.Frame(device_frame)
        device_list_frame.pack(fill='x', pady=5)
        
        ttk.Label(device_list_frame, text="Available Devices:", style='Info.TLabel').pack(anchor='w')
        
        # Device listbox
        self.device_listbox = tk.Listbox(device_list_frame, height=6, bg='#404040', fg='white', 
                                        selectbackground='#0066cc', font=('Arial', 10))
        self.device_listbox.pack(fill='x', pady=5)
        self.device_listbox.bind('<<ListboxSelect>>', self._on_device_select)
        
        # Device buttons
        device_buttons_frame = ttk.Frame(device_frame)
        device_buttons_frame.pack(fill='x', pady=5)
        
        refresh_btn = tk.Button(device_buttons_frame, text="🔄 Refresh Devices", 
                               command=self._detect_devices,
                               bg='#0066aa', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        refresh_btn.pack(side='left', padx=5)
        
        setup_btn = tk.Button(device_buttons_frame, text="⚙️ Setup Compression", 
                             command=self._setup_compression,
                             bg='#00aa00', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        setup_btn.pack(side='left', padx=5)
        
        # Compression status
        status_frame = ttk.LabelFrame(compression_frame, text="Compression Status", padding=10)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        self.compression_status_label = ttk.Label(status_frame, text="No device selected", style='Info.TLabel')
        self.compression_status_label.pack(anchor='w')
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(compression_frame, text="Compression Statistics", padding=10)
        stats_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Statistics text widget
        self.stats_text = tk.Text(stats_frame, height=10, bg='#404040', fg='white', 
                                 font=('Consolas', 9), wrap='word')
        self.stats_text.pack(fill='both', expand=True)
        
        # Action buttons
        action_frame = ttk.Frame(compression_frame)
        action_frame.pack(fill='x', padx=20, pady=10)
        
        auto_compress_btn = tk.Button(action_frame, text="🚀 Auto-Compress", 
                                     command=self._auto_compress,
                                     bg='#ff6600', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8)
        auto_compress_btn.pack(side='left', padx=5)
        
        suggestions_btn = tk.Button(action_frame, text="💡 Get Suggestions", 
                                   command=self._get_suggestions,
                                   bg='#6600ff', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8)
        suggestions_btn.pack(side='left', padx=5)
        
        test_btn = tk.Button(action_frame, text="🧪 Test Compression", 
                            command=self._test_compression,
                            bg='#006600', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8)
        test_btn.pack(side='left', padx=5)
    
    def _create_storage_tab(self):
        """Create the storage management tab."""
        storage_frame = ttk.Frame(self.notebook)
        self.notebook.add(storage_frame, text="💾 Storage Management")
        
        # Title
        title = ttk.Label(storage_frame, text="Multi-Device Storage Management", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(storage_frame, 
                        text="Manage multiple storage devices and configure compression settings\n"
                             "for optimal performance and space utilization.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=5)
        
        # Device overview
        overview_frame = ttk.LabelFrame(storage_frame, text="Device Overview", padding=10)
        overview_frame.pack(fill='x', padx=20, pady=10)
        
        self.device_overview_text = tk.Text(overview_frame, height=8, bg='#404040', fg='white', 
                                           font=('Consolas', 9), wrap='word')
        self.device_overview_text.pack(fill='both', expand=True)
        
        # Configuration options
        config_frame = ttk.LabelFrame(storage_frame, text="Configuration Options", padding=10)
        config_frame.pack(fill='x', padx=20, pady=10)
        
        # Compression threshold
        threshold_frame = ttk.Frame(config_frame)
        threshold_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_frame, text="Compression Threshold (%):", style='Info.TLabel').pack(side='left')
        self.threshold_var = tk.StringVar(value="50")
        threshold_entry = tk.Entry(threshold_frame, textvariable=self.threshold_var, width=10, 
                                  bg='#404040', fg='white')
        threshold_entry.pack(side='left', padx=10)
        
        # Pattern retention
        retention_frame = ttk.Frame(config_frame)
        retention_frame.pack(fill='x', pady=5)
        
        ttk.Label(retention_frame, text="Pattern Retention (days):", style='Info.TLabel').pack(side='left')
        self.retention_var = tk.StringVar(value="90")
        retention_entry = tk.Entry(retention_frame, textvariable=self.retention_var, width=10, 
                                  bg='#404040', fg='white')
        retention_entry.pack(side='left', padx=10)
        
        # Save configuration button
        save_config_btn = tk.Button(config_frame, text="💾 Save Configuration", 
                                   command=self._save_configuration,
                                   bg='#00aa00', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        save_config_btn.pack(pady=10)
    
    def _create_learning_tab(self):
        """Create the progressive learning configuration tab."""
        learning_frame = ttk.Frame(self.notebook)
        self.notebook.add(learning_frame, text="🧠 Progressive Learning")
        
        # Title
        title = ttk.Label(learning_frame, text="Progressive Learning Configuration", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(learning_frame, 
                        text="Configure how the system learns from trading patterns\n"
                             "and improves compression efficiency over time.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=5)
        
        # Learning options
        options_frame = ttk.LabelFrame(learning_frame, text="Learning Options", padding=10)
        options_frame.pack(fill='x', padx=20, pady=10)
        
        # Enable progressive learning
        self.learning_enabled_var = tk.BooleanVar(value=True)
        learning_check = tk.Checkbutton(options_frame, text="Enable Progressive Learning", 
                                       variable=self.learning_enabled_var,
                                       bg='#1e1e1e', fg='white', selectcolor='#404040',
                                       font=('Arial', 10))
        learning_check.pack(anchor='w', pady=5)
        
        # Learning interval
        interval_frame = ttk.Frame(options_frame)
        interval_frame.pack(fill='x', pady=5)
        
        ttk.Label(interval_frame, text="Learning Update Interval (hours):", style='Info.TLabel').pack(side='left')
        self.interval_var = tk.StringVar(value="1")
        interval_entry = tk.Entry(interval_frame, textvariable=self.interval_var, width=10, 
                                 bg='#404040', fg='white')
        interval_entry.pack(side='left', padx=10)
        
        # Weight matrices
        weights_frame = ttk.LabelFrame(learning_frame, text="Weight Matrix Configuration", padding=10)
        weights_frame.pack(fill='x', padx=20, pady=10)
        
        # Omega weight
        omega_frame = ttk.Frame(weights_frame)
        omega_frame.pack(fill='x', pady=5)
        
        ttk.Label(omega_frame, text="Ω (Omega) Weight:", style='Info.TLabel').pack(side='left')
        self.omega_weight_var = tk.StringVar(value="0.4")
        omega_entry = tk.Entry(omega_frame, textvariable=self.omega_weight_var, width=10, 
                              bg='#404040', fg='white')
        omega_entry.pack(side='left', padx=10)
        
        # Beta weight
        beta_frame = ttk.Frame(weights_frame)
        beta_frame.pack(fill='x', pady=5)
        
        ttk.Label(beta_frame, text="Β (Beta) Weight:", style='Info.TLabel').pack(side='left')
        self.beta_weight_var = tk.StringVar(value="0.3")
        beta_entry = tk.Entry(beta_frame, textvariable=self.beta_weight_var, width=10, 
                             bg='#404040', fg='white')
        beta_entry.pack(side='left', padx=10)
        
        # Gamma weight
        gamma_frame = ttk.Frame(weights_frame)
        gamma_frame.pack(fill='x', pady=5)
        
        ttk.Label(gamma_frame, text="Γ (Gamma) Weight:", style='Info.TLabel').pack(side='left')
        self.gamma_weight_var = tk.StringVar(value="0.3")
        gamma_entry = tk.Entry(gamma_frame, textvariable=self.gamma_weight_var, width=10, 
                              bg='#404040', fg='white')
        gamma_entry.pack(side='left', padx=10)
    
    def _create_scheduling_tab(self):
        """Create the advanced scheduling configuration tab."""
        scheduling_frame = ttk.Frame(self.notebook)
        self.notebook.add(scheduling_frame, text="⏰ Advanced Scheduling")
        
        # Title
        title = ttk.Label(scheduling_frame, text="Advanced Scheduling Configuration", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(scheduling_frame, 
                        text="Configure automated self-reconfiguration and storage optimization\n"
                             "during low-trading hours (1-4 AM) for optimal performance.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=5)
        
        # Create scrollable frame
        canvas = tk.Canvas(scheduling_frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(scheduling_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Timing Configuration
        timing_frame = ttk.LabelFrame(scrollable_frame, text="⏰ Low-Trading Hours Configuration", padding=10)
        timing_frame.pack(fill='x', padx=20, pady=10)
        
        # Low trading hours
        hours_frame = ttk.Frame(timing_frame)
        hours_frame.pack(fill='x', pady=5)
        
        ttk.Label(hours_frame, text="Low Trading Start Hour (1-4 AM):", style='Info.TLabel').pack(side='left')
        self.low_start_hour_var = tk.StringVar(value="1")
        start_hour_entry = tk.Entry(hours_frame, textvariable=self.low_start_hour_var, width=5, 
                                   bg='#404040', fg='white')
        start_hour_entry.pack(side='left', padx=10)
        
        ttk.Label(hours_frame, text="End Hour:", style='Info.TLabel').pack(side='left', padx=10)
        self.low_end_hour_var = tk.StringVar(value="4")
        end_hour_entry = tk.Entry(hours_frame, textvariable=self.low_end_hour_var, width=5, 
                                 bg='#404040', fg='white')
        end_hour_entry.pack(side='left', padx=10)
        
        # Preferred reconfiguration time
        pref_frame = ttk.Frame(timing_frame)
        pref_frame.pack(fill='x', pady=5)
        
        ttk.Label(pref_frame, text="Preferred Reconfiguration Time (1-4 AM):", style='Info.TLabel').pack(side='left')
        self.pref_reconfig_hour_var = tk.StringVar(value="2")
        pref_hour_entry = tk.Entry(pref_frame, textvariable=self.pref_reconfig_hour_var, width=5, 
                                  bg='#404040', fg='white')
        pref_hour_entry.pack(side='left', padx=10)
        
        # Compression Configuration
        compression_frame = ttk.LabelFrame(scrollable_frame, text="🔧 Compression Timing", padding=10)
        compression_frame.pack(fill='x', padx=20, pady=10)
        
        # Compression timeout
        timeout_frame = ttk.Frame(compression_frame)
        timeout_frame.pack(fill='x', pady=5)
        
        ttk.Label(timeout_frame, text="Compression Timeout (minutes):", style='Info.TLabel').pack(side='left')
        self.compression_timeout_var = tk.StringVar(value="30")
        timeout_entry = tk.Entry(timeout_frame, textvariable=self.compression_timeout_var, width=10, 
                                bg='#404040', fg='white')
        timeout_entry.pack(side='left', padx=10)
        
        # Retry attempts
        retry_frame = ttk.Frame(compression_frame)
        retry_frame.pack(fill='x', pady=5)
        
        ttk.Label(retry_frame, text="Compression Retry Attempts:", style='Info.TLabel').pack(side='left')
        self.retry_attempts_var = tk.StringVar(value="3")
        retry_entry = tk.Entry(retry_frame, textvariable=self.retry_attempts_var, width=10, 
                              bg='#404040', fg='white')
        retry_entry.pack(side='left', padx=10)
        
        # Registry Configuration
        registry_frame = ttk.LabelFrame(scrollable_frame, text="📋 Registry Synchronization", padding=10)
        registry_frame.pack(fill='x', padx=20, pady=10)
        
        # Sync interval
        sync_frame = ttk.Frame(registry_frame)
        sync_frame.pack(fill='x', pady=5)
        
        ttk.Label(sync_frame, text="Registry Sync Interval (hours):", style='Info.TLabel').pack(side='left')
        self.sync_interval_var = tk.StringVar(value="24")
        sync_entry = tk.Entry(sync_frame, textvariable=self.sync_interval_var, width=10, 
                             bg='#404040', fg='white')
        sync_entry.pack(side='left', padx=10)
        
        # Backup count
        backup_frame = ttk.Frame(registry_frame)
        backup_frame.pack(fill='x', pady=5)
        
        ttk.Label(backup_frame, text="Weight Matrix Backup Count (days):", style='Info.TLabel').pack(side='left')
        self.backup_count_var = tk.StringVar(value="7")
        backup_entry = tk.Entry(backup_frame, textvariable=self.backup_count_var, width=10, 
                               bg='#404040', fg='white')
        backup_entry.pack(side='left', padx=10)
        
        # Performance Monitoring
        perf_frame = ttk.LabelFrame(scrollable_frame, text="📊 Performance Monitoring", padding=10)
        perf_frame.pack(fill='x', padx=20, pady=10)
        
        # Check interval
        check_frame = ttk.Frame(perf_frame)
        check_frame.pack(fill='x', pady=5)
        
        ttk.Label(check_frame, text="Performance Check Interval (minutes):", style='Info.TLabel').pack(side='left')
        self.check_interval_var = tk.StringVar(value="60")
        check_entry = tk.Entry(check_frame, textvariable=self.check_interval_var, width=10, 
                              bg='#404040', fg='white')
        check_entry.pack(side='left', padx=10)
        
        # Drift threshold
        drift_frame = ttk.Frame(perf_frame)
        drift_frame.pack(fill='x', pady=5)
        
        ttk.Label(drift_frame, text="Drift Threshold (%):", style='Info.TLabel').pack(side='left')
        self.drift_threshold_var = tk.StringVar(value="5")
        drift_entry = tk.Entry(drift_frame, textvariable=self.drift_threshold_var, width=10, 
                              bg='#404040', fg='white')
        drift_entry.pack(side='left', padx=10)
        
        # Feature Toggles
        features_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Feature Configuration", padding=10)
        features_frame.pack(fill='x', padx=20, pady=10)
        
        # Auto-compression
        self.auto_compression_var = tk.BooleanVar(value=True)
        auto_comp_check = tk.Checkbutton(features_frame, text="Enable Auto-Compression", 
                                        variable=self.auto_compression_var,
                                        bg='#1e1e1e', fg='white', selectcolor='#404040',
                                        font=('Arial', 10))
        auto_comp_check.pack(anchor='w', pady=5)
        
        # Storage optimization
        self.storage_optimization_var = tk.BooleanVar(value=True)
        storage_opt_check = tk.Checkbutton(features_frame, text="Enable Storage Optimization", 
                                          variable=self.storage_optimization_var,
                                          bg='#1e1e1e', fg='white', selectcolor='#404040',
                                          font=('Arial', 10))
        storage_opt_check.pack(anchor='w', pady=5)
        
        # Backup rotation
        self.backup_rotation_var = tk.BooleanVar(value=True)
        backup_rot_check = tk.Checkbutton(features_frame, text="Enable Backup Rotation", 
                                         variable=self.backup_rotation_var,
                                         bg='#1e1e1e', fg='white', selectcolor='#404040',
                                         font=('Arial', 10))
        backup_rot_check.pack(anchor='w', pady=5)
        
        # Multi-device sync
        self.multi_device_sync_var = tk.BooleanVar(value=True)
        multi_sync_check = tk.Checkbutton(features_frame, text="Enable Multi-Device Synchronization", 
                                         variable=self.multi_device_sync_var,
                                         bg='#1e1e1e', fg='white', selectcolor='#404040',
                                         font=('Arial', 10))
        multi_sync_check.pack(anchor='w', pady=5)
        
        # API monitoring
        self.api_monitoring_var = tk.BooleanVar(value=True)
        api_mon_check = tk.Checkbutton(features_frame, text="Enable API Connectivity Monitoring", 
                                      variable=self.api_monitoring_var,
                                      bg='#1e1e1e', fg='white', selectcolor='#404040',
                                      font=('Arial', 10))
        api_mon_check.pack(anchor='w', pady=5)
        
        # Memory optimization
        self.memory_optimization_var = tk.BooleanVar(value=True)
        memory_opt_check = tk.Checkbutton(features_frame, text="Enable Memory Optimization", 
                                         variable=self.memory_optimization_var,
                                         bg='#1e1e1e', fg='white', selectcolor='#404040',
                                         font=('Arial', 10))
        memory_opt_check.pack(anchor='w', pady=5)
        
        # Scheduler Status
        status_frame = ttk.LabelFrame(scrollable_frame, text="📊 Scheduler Status", padding=10)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        # Status display
        self.scheduler_status_text = tk.Text(status_frame, height=8, bg='#404040', fg='white', 
                                            font=('Consolas', 9), wrap='word')
        self.scheduler_status_text.pack(fill='both', expand=True)
        
        # Action buttons
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill='x', padx=20, pady=10)
        
        start_scheduler_btn = tk.Button(action_frame, text="🚀 Start Scheduler", 
                                       command=self._start_scheduler,
                                       bg='#00aa00', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8)
        start_scheduler_btn.pack(side='left', padx=5)
        
        stop_scheduler_btn = tk.Button(action_frame, text="🛑 Stop Scheduler", 
                                      command=self._stop_scheduler,
                                      bg='#aa0000', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8)
        stop_scheduler_btn.pack(side='left', padx=5)
        
        refresh_status_btn = tk.Button(action_frame, text="🔄 Refresh Status", 
                                      command=self._refresh_scheduler_status,
                                      bg='#0066aa', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8)
        refresh_status_btn.pack(side='left', padx=5)
        
        save_config_btn = tk.Button(action_frame, text="💾 Save Configuration", 
                                   command=self._save_scheduling,
                                   bg='#6600ff', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=8)
        save_config_btn.pack(side='left', padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Load existing configuration
        self._load_scheduling_config()
        
        # Initial status update
        self._refresh_scheduler_status()
    
    def _create_education_tab(self):
        """Create the educational content tab."""
        education_frame = ttk.Frame(self.notebook)
        self.notebook.add(education_frame, text="📚 How It Works")
        
        # Title
        title = ttk.Label(education_frame, text="How Alpha Encryption Compression Works", style='Header.TLabel')
        title.pack(pady=10)
        
        # Educational content
        content_frame = ttk.Frame(education_frame)
        content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create scrollable text widget
        canvas = tk.Canvas(content_frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Educational content
        education_text = """
🔐 Alpha Encryption-Based Intelligent Compression

What is Alpha Encryption?
Alpha Encryption uses a sophisticated mathematical system called Ω-B-Γ Logic (Omega-Beta-Gamma Logic) 
developed by Maxamillion M.A.A. DeLeon ("The Schwa") & Nexus AI. This system provides:

Ω (Omega) Layer: Recursive mathematical operations with complex state management
Β (Beta) Layer: Quantum-inspired logic gates with Bayesian entropy  
Γ (Gamma) Layer: Harmonic frequency analysis with wave entropy

How Does Intelligent Compression Work?

1. PATTERN RECOGNITION
   - The system analyzes your trading data using Ω-B-Γ Logic
   - Identifies recurring patterns in price movements, indicators, and strategies
   - Creates mathematical representations of successful trading patterns

2. WEIGHT MATRIX GENERATION
   - Converts trading patterns into encrypted weight matrices
   - Each pattern type (backtest, live_trade, market_data) gets its own weight matrix
   - These matrices contain the "essence" of your trading knowledge

3. PROGRESSIVE LEARNING
   - As you trade more, the system learns from successful patterns
   - Updates weight matrices to improve future compression and trading decisions
   - Builds a "memory" of what works in different market conditions

4. INTELLIGENT STORAGE OPTIMIZATION
   - Monitors storage usage and automatically triggers compression when needed
   - Converts old, raw data into compressed weight states
   - Maintains a registry for instant pattern recall

Why This is Revolutionary:

🎯 BETTER TRADING PERFORMANCE
   - The system doesn't just compress data, it learns from it
   - Weight matrices can be used to improve trading decisions
   - Progressive learning means the system gets smarter over time

💾 INTELLIGENT STORAGE MANAGEMENT
   - Automatic compression when storage reaches 50% usage
   - Can extend your 23GB USB capacity by 2-3x through intelligent compression
   - Maintains data integrity while maximizing space efficiency

🔒 SECURE & PORTABLE
   - All data is encrypted using military-grade Alpha Encryption
   - Your API keys and trading data are never stored on your main device
   - Perfect for portable trading with USB drives

📈 SCALABLE LEARNING
   - The more you trade, the better the system becomes
   - Weight matrices evolve based on market performance
   - Creates a personalized trading knowledge base

Storage Capacity Analysis for 23GB USB:

Conservative Usage (Recommended for 1+ years):
- Live Trading: ~2.2GB/year (6MB/day)
- Weekly Backtesting: ~2.6GB/year (50MB/week)  
- System Overhead: ~1GB/year
- Total: ~5.8GB/year ✅ 4+ years capacity!

With Intelligent Compression:
- Can extend capacity by 2-3x through pattern compression
- Progressive learning reduces redundant data storage
- Automatic optimization based on usage patterns

Setup Recommendations:

1. ENABLE ALPHA COMPRESSION
   - Select your USB drive or preferred storage device
   - Click "Setup Compression" to initialize the system
   - The system will create necessary directories and test compression

2. CONFIGURE LEARNING PARAMETERS
   - Set compression threshold (default: 50%)
   - Configure pattern retention period (default: 90 days)
   - Adjust weight matrix parameters if needed

3. MONITOR & OPTIMIZE
   - Use "Get Suggestions" to see optimization opportunities
   - Run "Auto-Compress" when storage usage is high
   - Review compression statistics regularly

The system is designed to be:
- AUTOMATIC: Requires minimal user intervention
- INTELLIGENT: Learns and improves over time
- SECURE: Military-grade encryption for all data
- PORTABLE: Works with any storage device
- SCALABLE: Grows with your trading needs

This isn't just compression - it's intelligent data management that makes your trading system better over time!
        """
        
        # Create text widget for educational content
        edu_text_widget = tk.Text(scrollable_frame, bg='#404040', fg='white', 
                                 font=('Arial', 10), wrap='word', padx=20, pady=20)
        edu_text_widget.insert('1.0', education_text)
        edu_text_widget.config(state='disabled')  # Make read-only
        edu_text_widget.pack(fill='both', expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_settings_tab(self):
        """Create the advanced settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Advanced Settings")
        
        # Title
        title = ttk.Label(settings_frame, text="Advanced Configuration", style='Header.TLabel')
        title.pack(pady=10)
        
        # Settings content
        settings_content = ttk.LabelFrame(settings_frame, text="System Settings", padding=10)
        settings_content.pack(fill='x', padx=20, pady=10)
        
        # Auto-compression settings
        auto_frame = ttk.LabelFrame(settings_content, text="Auto-Compression Settings", padding=10)
        auto_frame.pack(fill='x', pady=5)
        
        self.auto_compression_var = tk.BooleanVar(value=True)
        auto_check = tk.Checkbutton(auto_frame, text="Enable Auto-Compression", 
                                   variable=self.auto_compression_var,
                                   bg='#1e1e1e', fg='white', selectcolor='#404040',
                                   font=('Arial', 10))
        auto_check.pack(anchor='w', pady=5)
        
        # Notification settings
        self.notifications_var = tk.BooleanVar(value=True)
        notif_check = tk.Checkbutton(auto_frame, text="Enable Compression Notifications", 
                                    variable=self.notifications_var,
                                    bg='#1e1e1e', fg='white', selectcolor='#404040',
                                    font=('Arial', 10))
        notif_check.pack(anchor='w', pady=5)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(settings_content, text="Performance Settings", padding=10)
        perf_frame.pack(fill='x', pady=5)
        
        # Max patterns per type
        max_patterns_frame = ttk.Frame(perf_frame)
        max_patterns_frame.pack(fill='x', pady=5)
        
        ttk.Label(max_patterns_frame, text="Max Patterns per Type:", style='Info.TLabel').pack(side='left')
        self.max_patterns_var = tk.StringVar(value="1000")
        max_patterns_entry = tk.Entry(max_patterns_frame, textvariable=self.max_patterns_var, width=10, 
                                     bg='#404040', fg='white')
        max_patterns_entry.pack(side='left', padx=10)
        
        # Save settings button
        save_settings_btn = tk.Button(settings_content, text="💾 Save Settings", 
                                     command=self._save_settings,
                                     bg='#00aa00', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        save_settings_btn.pack(pady=10)
    
    def _create_status_bar(self):
        """Create the status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom', padx=20, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Info.TLabel')
        self.status_label.pack(side='left')
        
        # Version info
        version_label = ttk.Label(status_frame, text="Alpha Encryption v1.0", style='Info.TLabel')
        version_label.pack(side='right')
    
    def _detect_devices(self):
        """Detect available storage devices."""
        if not COMPRESSION_AVAILABLE:
            self._update_status("Alpha Compression not available")
            return
        
        def detect_thread():
            self._update_status("Detecting storage devices...")
            
            try:
                devices = self.device_manager.detect_available_devices()
                
                # Update device listbox
                self.device_listbox.delete(0, tk.END)
                
                for device in devices:
                    device_info = f"{device.device_name} ({device.device_type}) - {device.free_space / (1024**3):.1f}GB free"
                    if device.compression_enabled:
                        device_info += " [Compression Enabled]"
                    
                    self.device_listbox.insert(tk.END, device_info)
                
                self._update_status(f"Detected {len(devices)} storage devices")
                
            except Exception as e:
                self._update_status(f"Error detecting devices: {e}")
        
        threading.Thread(target=detect_thread, daemon=True).start()
    
    def _on_device_select(self, event):
        """Handle device selection."""
        if not self.device_listbox.curselection():
            return
        
        selection = self.device_listbox.curselection()[0]
        device_info = self.device_listbox.get(selection)
        
        # Extract device path from selection
        devices = self.device_manager.detect_available_devices()
        if selection < len(devices):
            self.selected_device = devices[selection]
            self._update_device_status()
    
    def _update_device_status(self):
        """Update device status display."""
        if not self.selected_device:
            self.compression_status_label.config(text="No device selected")
            return
        
        device = self.selected_device
        status_text = f"Device: {device.device_name}\n"
        status_text += f"Type: {device.device_type}\n"
        status_text += f"Free Space: {device.free_space / (1024**3):.1f}GB\n"
        status_text += f"Compression: {'Enabled' if device.compression_enabled else 'Disabled'}"
        
        self.compression_status_label.config(text=status_text)
        
        # Update statistics if compression is enabled
        if device.compression_enabled:
            self._update_compression_stats()
    
    def _update_compression_stats(self):
        """Update compression statistics display."""
        if not self.selected_device or not self.selected_device.compression_enabled:
            return
        
        try:
            stats = self.device_manager.get_device_compression_stats(self.selected_device.device_path)
            if stats:
                stats_text = "=== COMPRESSION STATISTICS ===\n\n"
                
                # Storage metrics
                storage = stats['storage_metrics']
                stats_text += f"Storage Usage: {storage['usage_ratio']:.1%}\n"
                stats_text += f"Free Space: {storage['free_space'] / (1024**3):.1f}GB\n"
                stats_text += f"Compression Threshold: {storage['compression_threshold']:.1%}\n\n"
                
                # Compression stats
                compression = stats['compression_stats']
                stats_text += f"Total Patterns: {compression['total_patterns']}\n"
                stats_text += f"Patterns Compressed: {compression['patterns_compressed']}\n"
                stats_text += f"Space Saved: {compression['space_saved'] / (1024**2):.1f}MB\n\n"
                
                # Type statistics
                stats_text += "=== PATTERN TYPE STATISTICS ===\n"
                for pattern_type, type_stats in stats['type_statistics'].items():
                    stats_text += f"\n{pattern_type.upper()}:\n"
                    stats_text += f"  Count: {type_stats['count']}\n"
                    stats_text += f"  Avg Compression: {type_stats['avg_compression_ratio']:.1%}\n"
                    stats_text += f"  Total Original: {type_stats['total_original'] / (1024**2):.1f}MB\n"
                    stats_text += f"  Total Compressed: {type_stats['total_compressed'] / (1024**2):.1f}MB\n"
                
                self.stats_text.delete('1.0', tk.END)
                self.stats_text.insert('1.0', stats_text)
            
        except Exception as e:
            self.stats_text.delete('1.0', tk.END)
            self.stats_text.insert('1.0', f"Error loading statistics: {e}")
    
    def _setup_compression(self):
        """Setup compression on selected device."""
        if not self.selected_device:
            messagebox.showwarning("No Device Selected", "Please select a storage device first.")
            return
        
        def setup_thread():
            self._update_status(f"Setting up Alpha compression on {self.selected_device.device_name}...")
            
            try:
                success = self.device_manager.setup_compression_on_device(self.selected_device.device_path)
                
                if success:
                    self._update_status(f"✅ Alpha compression setup successful on {self.selected_device.device_name}")
                    messagebox.showinfo("Setup Complete", 
                                      f"Alpha compression has been successfully set up on {self.selected_device.device_name}!\n\n"
                                      "The system is now ready to intelligently compress your trading data.")
                    
                    # Refresh device list
                    self._detect_devices()
                else:
                    self._update_status(f"❌ Alpha compression setup failed on {self.selected_device.device_name}")
                    messagebox.showerror("Setup Failed", 
                                       f"Failed to set up Alpha compression on {self.selected_device.device_name}.\n"
                                       "Please check device permissions and try again.")
            
            except Exception as e:
                self._update_status(f"Error during setup: {e}")
                messagebox.showerror("Setup Error", f"An error occurred during setup: {e}")
        
        threading.Thread(target=setup_thread, daemon=True).start()
    
    def _auto_compress(self):
        """Run auto-compression on selected device."""
        if not self.selected_device or not self.selected_device.compression_enabled:
            messagebox.showwarning("Compression Not Available", 
                                 "Please select a device with compression enabled first.")
            return
        
        def compress_thread():
            self._update_status("Running auto-compression...")
            
            try:
                result = auto_compress_device_data(self.selected_device.device_path)
                
                self._update_status(f"Auto-compression complete: {result['message']}")
                
                if result['compressed'] > 0:
                    messagebox.showinfo("Compression Complete", 
                                      f"Successfully compressed {result['compressed']} patterns!\n"
                                      f"Space saved: {result['space_saved'] / (1024**2):.1f}MB")
                else:
                    messagebox.showinfo("No Compression Needed", result['message'])
                
                # Update statistics
                self._update_compression_stats()
            
            except Exception as e:
                self._update_status(f"Error during compression: {e}")
                messagebox.showerror("Compression Error", f"An error occurred: {e}")
        
        threading.Thread(target=compress_thread, daemon=True).start()
    
    def _get_suggestions(self):
        """Get compression optimization suggestions."""
        if not self.selected_device or not self.selected_device.compression_enabled:
            messagebox.showwarning("Compression Not Available", 
                                 "Please select a device with compression enabled first.")
            return
        
        def suggestions_thread():
            self._update_status("Analyzing compression opportunities...")
            
            try:
                suggestions = get_device_compression_suggestions(self.selected_device.device_path)
                
                if suggestions['suggestions']:
                    suggestion_text = "=== COMPRESSION OPTIMIZATION SUGGESTIONS ===\n\n"
                    suggestion_text += f"Total Suggestions: {suggestions['total_suggestions']}\n"
                    suggestion_text += f"Estimated Total Savings: {suggestions['estimated_total_savings']}\n\n"
                    
                    for i, suggestion in enumerate(suggestions['suggestions'], 1):
                        suggestion_text += f"{i}. {suggestion['message']}\n"
                        suggestion_text += f"   Priority: {suggestion['priority'].upper()}\n"
                        suggestion_text += f"   Estimated Savings: {suggestion['estimated_savings']}\n\n"
                    
                    # Show in a new window
                    self._show_suggestions_window(suggestion_text)
                else:
                    messagebox.showinfo("No Suggestions", "No compression optimization suggestions at this time.")
                
                self._update_status("Analysis complete")
            
            except Exception as e:
                self._update_status(f"Error getting suggestions: {e}")
                messagebox.showerror("Analysis Error", f"An error occurred: {e}")
        
        threading.Thread(target=suggestions_thread, daemon=True).start()
    
    def _show_suggestions_window(self, suggestions_text):
        """Show suggestions in a new window."""
        suggestions_window = tk.Toplevel(self.root)
        suggestions_window.title("💡 Compression Optimization Suggestions")
        suggestions_window.geometry("600x500")
        suggestions_window.configure(bg='#1e1e1e')
        
        # Title
        title = ttk.Label(suggestions_window, text="Optimization Suggestions", style='Header.TLabel')
        title.pack(pady=10)
        
        # Text widget
        text_widget = tk.Text(suggestions_window, bg='#404040', fg='white', 
                             font=('Consolas', 10), wrap='word', padx=20, pady=20)
        text_widget.insert('1.0', suggestions_text)
        text_widget.config(state='disabled')
        text_widget.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Close button
        close_btn = tk.Button(suggestions_window, text="Close", 
                             command=suggestions_window.destroy,
                             bg='#0066aa', fg='white', font=('Arial', 10, 'bold'), padx=20, pady=5)
        close_btn.pack(pady=10)
    
    def _test_compression(self):
        """Test compression with sample data."""
        if not self.selected_device or not self.selected_device.compression_enabled:
            messagebox.showwarning("Compression Not Available", 
                                 "Please select a device with compression enabled first.")
            return
        
        def test_thread():
            self._update_status("Testing compression with sample data...")
            
            try:
                # Create sample trading data
                sample_data = {
                    'timestamp': time.time(),
                    'symbol': 'BTC/USDC',
                    'price': 45000.0,
                    'volume': 100.0,
                    'indicators': {
                        'rsi': 65.5,
                        'macd': 0.0023,
                        'bollinger_bands': [44000, 45000, 46000]
                    },
                    'strategy': 'momentum_based',
                    'confidence': 0.85,
                    'test_data': True
                }
                
                # Compress sample data
                compressed = compress_trading_data_on_device(
                    self.selected_device.device_path, sample_data, 'test'
                )
                
                if compressed:
                    self._update_status(f"✅ Test compression successful: {compressed.compression_ratio:.1%} ratio")
                    messagebox.showinfo("Test Successful", 
                                      f"Compression test completed successfully!\n\n"
                                      f"Compression Ratio: {compressed.compression_ratio:.1%}\n"
                                      f"Original Size: {compressed.original_size} bytes\n"
                                      f"Compressed Size: {compressed.compressed_size} bytes\n"
                                      f"Space Saved: {compressed.original_size - compressed.compressed_size} bytes")
                    
                    # Update statistics
                    self._update_compression_stats()
                else:
                    self._update_status("❌ Test compression failed")
                    messagebox.showerror("Test Failed", "Compression test failed. Please check system logs.")
            
            except Exception as e:
                self._update_status(f"Error during test: {e}")
                messagebox.showerror("Test Error", f"An error occurred during testing: {e}")
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def _save_configuration(self):
        """Save compression configuration."""
        try:
            # Get configuration values
            config = {
                'compression_threshold': float(self.threshold_var.get()) / 100,
                'pattern_retention_days': int(self.retention_var.get()),
                'auto_compression_enabled': self.auto_compression_var.get(),
                'notifications_enabled': self.notifications_var.get(),
                'max_patterns_per_type': int(self.max_patterns_var.get()),
                'progressive_learning_enabled': self.learning_enabled_var.get(),
                'learning_update_interval': int(self.interval_var.get()) * 3600,
                'alpha_config': {
                    'omega_weight': float(self.omega_weight_var.get()),
                    'beta_weight': float(self.beta_weight_var.get()),
                    'gamma_weight': float(self.gamma_weight_var.get())
                }
            }
            
            # Save to file
            config_file = Path("AOI_Base_Files_Schwabot/config/compression_config.json")
            os.makedirs(config_file.parent, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self._update_status("✅ Configuration saved successfully")
            messagebox.showinfo("Configuration Saved", "Compression configuration has been saved successfully!")
            
        except Exception as e:
            self._update_status(f"Error saving configuration: {e}")
            messagebox.showerror("Save Error", f"Failed to save configuration: {e}")
    
    def _save_settings(self):
        """Save advanced settings."""
        try:
            settings = {
                'auto_compression_enabled': self.auto_compression_var.get(),
                'notifications_enabled': self.notifications_var.get(),
                'max_patterns_per_type': int(self.max_patterns_var.get())
            }
            
            # Save to file
            settings_file = Path("AOI_Base_Files_Schwabot/config/advanced_settings.json")
            os.makedirs(settings_file.parent, exist_ok=True)
            
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            self._update_status("✅ Settings saved successfully")
            messagebox.showinfo("Settings Saved", "Advanced settings have been saved successfully!")
            
        except Exception as e:
            self._update_status(f"Error saving settings: {e}")
            messagebox.showerror("Save Error", f"Failed to save settings: {e}")
    
    def _save_scheduling(self):
        """Save advanced scheduling settings."""
        try:
            scheduling_config = {
                'auto_compression_enabled': self.auto_compression_var.get(),
                'storage_optimization_enabled': self.storage_optimization_var.get(),
                'backup_rotation_enabled': self.backup_rotation_var.get(),
                'multi_device_sync_enabled': self.multi_device_sync_var.get(),
                'api_monitoring_enabled': self.api_monitoring_var.get(),
                'memory_optimization_enabled': self.memory_optimization_var.get(),
                'low_trading_hours': {
                    'start_hour': int(self.low_start_hour_var.get()),
                    'end_hour': int(self.low_end_hour_var.get())
                },
                'preferred_reconfiguration_hour': int(self.pref_reconfig_hour_var.get()),
                'compression_timeout': int(self.compression_timeout_var.get()),
                'compression_retry_attempts': int(self.retry_attempts_var.get()),
                'registry_sync_interval': int(self.sync_interval_var.get()) * 3600,
                'weight_matrix_backup_count': int(self.backup_count_var.get()),
                'performance_check_interval': int(self.check_interval_var.get()) * 60,
                'drift_threshold': float(self.drift_threshold_var.get()) / 100
            }
            
            # Save to file
            scheduling_file = Path("AOI_Base_Files_Schwabot/config/scheduling_config.json")
            os.makedirs(scheduling_file.parent, exist_ok=True)
            
            with open(scheduling_file, 'w') as f:
                json.dump(scheduling_config, f, indent=2)
            
            self._update_status("✅ Scheduling configuration saved successfully")
            messagebox.showinfo("Scheduling Saved", "Advanced scheduling configuration has been saved successfully!")
            
        except Exception as e:
            self._update_status(f"Error saving scheduling: {e}")
            messagebox.showerror("Save Error", f"Failed to save scheduling configuration: {e}")
    
    def _load_scheduling_config(self):
        """Load existing scheduling configuration."""
        try:
            scheduling_file = Path("AOI_Base_Files_Schwabot/config/scheduling_config.json")
            if scheduling_file.exists():
                with open(scheduling_file, 'r') as f:
                    config = json.load(f)
                
                # Update UI with loaded configuration
                self.auto_compression_var.set(config.get('auto_compression_enabled', True))
                self.storage_optimization_var.set(config.get('storage_optimization_enabled', True))
                self.backup_rotation_var.set(config.get('backup_rotation_enabled', True))
                self.multi_device_sync_var.set(config.get('multi_device_sync_enabled', True))
                self.api_monitoring_var.set(config.get('api_monitoring_enabled', True))
                self.memory_optimization_var.set(config.get('memory_optimization_enabled', True))
                
                # Load timing configuration
                low_trading_hours = config.get('low_trading_hours', {'start_hour': 1, 'end_hour': 4})
                self.low_start_hour_var.set(str(low_trading_hours.get('start_hour', 1)))
                self.low_end_hour_var.set(str(low_trading_hours.get('end_hour', 4)))
                
                self.pref_reconfig_hour_var.set(str(config.get('preferred_reconfiguration_hour', 2)))
                self.compression_timeout_var.set(str(config.get('compression_timeout', 30)))
                self.retry_attempts_var.set(str(config.get('compression_retry_attempts', 3)))
                self.sync_interval_var.set(str(config.get('registry_sync_interval', 24 * 3600) // 3600))
                self.backup_count_var.set(str(config.get('weight_matrix_backup_count', 7)))
                self.check_interval_var.set(str(config.get('performance_check_interval', 60 * 60) // 60))
                self.drift_threshold_var.set(str(int(config.get('drift_threshold', 0.05) * 100)))
                
        except Exception as e:
            self.logger.warning(f"Failed to load scheduling configuration: {e}")
    
    def _start_scheduler(self):
        """Start the advanced scheduler."""
        try:
            # Import and start scheduler
            from advanced_scheduler import start_advanced_scheduler
            
            start_advanced_scheduler()
            
            self._update_status("✅ Advanced scheduler started successfully")
            messagebox.showinfo("Scheduler Started", 
                              "Advanced scheduler has been started successfully!\n\n"
                              "The system will now automatically reconfigure during low-trading hours (1-4 AM).")
            
            # Refresh status
            self._refresh_scheduler_status()
            
        except Exception as e:
            self._update_status(f"Error starting scheduler: {e}")
            messagebox.showerror("Start Error", f"Failed to start scheduler: {e}")
    
    def _stop_scheduler(self):
        """Stop the advanced scheduler."""
        try:
            # Import and stop scheduler
            from advanced_scheduler import stop_advanced_scheduler
            
            stop_advanced_scheduler()
            
            self._update_status("🛑 Advanced scheduler stopped")
            messagebox.showinfo("Scheduler Stopped", 
                              "Advanced scheduler has been stopped successfully.")
            
            # Refresh status
            self._refresh_scheduler_status()
            
        except Exception as e:
            self._update_status(f"Error stopping scheduler: {e}")
            messagebox.showerror("Stop Error", f"Failed to stop scheduler: {e}")
    
    def _refresh_scheduler_status(self):
        """Refresh scheduler status display."""
        try:
            # Import scheduler
            from advanced_scheduler import get_advanced_scheduler
            
            scheduler = get_advanced_scheduler()
            status = scheduler.get_scheduler_status()
            
            # Format status display
            status_text = "=== ADVANCED SCHEDULER STATUS ===\n\n"
            
            # Basic status
            status_text += f"Status: {'🟢 Running' if status['is_running'] else '🔴 Stopped'}\n"
            status_text += f"Next Reconfiguration: {scheduler.get_next_scheduled_time()}\n\n"
            
            # Timing information
            if status['last_reconfig_time'] > 0:
                last_reconfig = datetime.fromtimestamp(status['last_reconfig_time']).strftime('%Y-%m-%d %H:%M:%S')
                status_text += f"Last Reconfiguration: {last_reconfig}\n"
            else:
                status_text += "Last Reconfiguration: Never\n"
            
            if status['last_compression_time'] > 0:
                last_compression = datetime.fromtimestamp(status['last_compression_time']).strftime('%Y-%m-%d %H:%M:%S')
                status_text += f"Last Compression: {last_compression}\n"
            else:
                status_text += "Last Compression: Never\n"
            
            if status['last_sync_time'] > 0:
                last_sync = datetime.fromtimestamp(status['last_sync_time']).strftime('%Y-%m-%d %H:%M:%S')
                status_text += f"Last Sync: {last_sync}\n"
            else:
                status_text += "Last Sync: Never\n"
            
            status_text += "\n"
            
            # Statistics
            status_text += f"Registry Entries: {status['registry_entries']}\n"
            status_text += f"Daily Weights: {status['daily_weights_count']}\n"
            status_text += f"Performance History: {status['performance_history_count']} entries\n\n"
            
            # Configuration summary
            config = status['config']
            status_text += "=== CONFIGURATION SUMMARY ===\n"
            status_text += f"Low Trading Hours: {config['low_trading_start_hour']:02d}:00 - {config['low_trading_end_hour']:02d}:00\n"
            status_text += f"Preferred Reconfig: {config['preferred_reconfig_hour']:02d}:00\n"
            status_text += f"Performance Check: Every {config['performance_check_interval_minutes']} minutes\n"
            status_text += f"Registry Sync: Every {config['registry_sync_interval_hours']} hours\n"
            status_text += f"Drift Threshold: {config['drift_threshold']:.1%}\n"
            
            # Update status text
            self.scheduler_status_text.delete('1.0', tk.END)
            self.scheduler_status_text.insert('1.0', status_text)
            
        except Exception as e:
            error_text = f"Error refreshing scheduler status: {e}"
            self.scheduler_status_text.delete('1.0', tk.END)
            self.scheduler_status_text.insert('1.0', error_text)
            self._update_status(f"Error refreshing status: {e}")
    
    def _update_status(self, message: str):
        """Update status bar message."""
        if self.status_label:
            self.status_label.config(text=message)


def show_advanced_options(parent=None):
    """Show the Advanced Options GUI."""
    gui = AdvancedOptionsGUI(parent)
    gui.show_advanced_options()


if __name__ == "__main__":
    show_advanced_options() 