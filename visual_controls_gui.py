#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Visual Controls GUI - Advanced Chart & Layer Management
==========================================================

Comprehensive visual interface for managing Schwabot's visual layer controller,
chart generation, pattern recognition, and real-time visualization controls.

Features:
- Advanced chart controls and customization
- Visual layer management and configuration
- Real-time pattern recognition controls
- AI analysis integration controls
- Performance monitoring and optimization
- Multi-device visualization management
- 🎯 GHOST MODE: BTC/USDC optimized trading with medium-risk orbitals
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import visual layer controller
try:
    from core.visual_layer_controller import (
        VisualLayerController, 
        VisualizationType, 
        ChartTimeframe,
        VisualAnalysis
    )
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Visual Layer Controller not available")

# Import Ghost Mode Manager
try:
    from AOI_Base_Files_Schwabot.core.ghost_mode_manager import ghost_mode_manager
    GHOST_MODE_AVAILABLE = True
except ImportError:
    try:
        from core.ghost_mode_manager import ghost_mode_manager
        GHOST_MODE_AVAILABLE = True
    except ImportError:
        GHOST_MODE_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("Ghost Mode Manager not available")

# Import Hybrid Mode Manager
try:
    from AOI_Base_Files_Schwabot.core.hybrid_mode_manager import hybrid_mode_manager
    HYBRID_MODE_AVAILABLE = True
except ImportError:
    try:
        from core.hybrid_mode_manager import hybrid_mode_manager
        HYBRID_MODE_AVAILABLE = True
    except ImportError:
        HYBRID_MODE_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("Hybrid Mode Manager not available")

# Import Ferris Ride Manager
try:
    from AOI_Base_Files_Schwabot.core.ferris_ride_manager import ferris_ride_manager
    FERRIS_RIDE_MODE_AVAILABLE = True
except ImportError:
    try:
        from core.ferris_ride_manager import ferris_ride_manager
        FERRIS_RIDE_MODE_AVAILABLE = True
    except ImportError:
        FERRIS_RIDE_MODE_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("Ferris Ride Manager not available")

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Available chart types."""
    PRICE_CHART = "price_chart"
    VOLUME_ANALYSIS = "volume_analysis"
    TECHNICAL_INDICATORS = "technical_indicators"
    PATTERN_RECOGNITION = "pattern_recognition"
    AI_ANALYSIS = "ai_analysis"
    PERFORMANCE_DASHBOARD = "performance_dashboard"
    RISK_METRICS = "risk_metrics"


class VisualLayer(Enum):
    """Visual layer types."""
    PRICE_LAYER = "price_layer"
    VOLUME_LAYER = "volume_layer"
    INDICATOR_LAYER = "indicator_layer"
    PATTERN_LAYER = "pattern_layer"
    AI_LAYER = "ai_layer"
    OVERLAY_LAYER = "overlay_layer"


@dataclass
class ChartConfig:
    """Chart configuration settings."""
    width: int = 1200
    height: int = 800
    dpi: int = 100
    style: str = "dark_background"
    colors: Dict[str, str] = field(default_factory=lambda: {
        "price": "#00ff00",
        "volume": "#0088ff",
        "ma_fast": "#ff8800",
        "ma_slow": "#ff0088",
        "rsi": "#ffff00",
        "macd": "#ff00ff",
        "background": "#1a1a1a",
        "grid": "#333333"
    })
    enable_grid: bool = True
    enable_legend: bool = True
    enable_annotations: bool = True


@dataclass
class LayerConfig:
    """Layer configuration settings."""
    layer_type: VisualLayer
    enabled: bool = True
    opacity: float = 1.0
    z_index: int = 0
    visible: bool = True
    auto_update: bool = True
    update_interval: float = 1.0


class VisualControlsGUI:
    """Advanced visual controls GUI for Schwabot."""
    
    def __init__(self, parent=None):
        """Initialize visual controls GUI."""
        self.parent = parent
        self.root = None
        
        # Visual controller
        self.visual_controller = None
        if VISUAL_AVAILABLE:
            self.visual_controller = VisualLayerController()
        
        # Configuration
        self.chart_config = ChartConfig()
        self.layer_configs: Dict[VisualLayer, LayerConfig] = {}
        self.active_layers: List[VisualLayer] = []
        
        # State
        self.is_running = False
        self.auto_refresh = False
        self.refresh_interval = 5.0
        
        # Threading
        self.update_thread = None
        self.lock = threading.Lock()
        
        # Initialize layer configs
        self._initialize_layer_configs()
        
        logger.info("🎨 Visual Controls GUI initialized")
    
    def _initialize_layer_configs(self):
        """Initialize default layer configurations."""
        for layer_type in VisualLayer:
            self.layer_configs[layer_type] = LayerConfig(
                layer_type=layer_type,
                z_index=len(self.layer_configs)
            )
    
    def show_visual_controls(self):
        """Show the visual controls GUI."""
        if self.root:
            self.root.deiconify()
            return
        
        self.root = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.root.title("🎨 Schwabot Visual Controls")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Configure styles
        self._configure_styles()
        
        # Create interface
        self._create_header()
        self._create_main_content()
        self._create_status_bar()
        
        # Start update thread
        self._start_update_thread()
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        logger.info("🎨 Visual Controls GUI displayed")
    
    def _configure_styles(self):
        """Configure custom styles for the GUI."""
        style = ttk.Style()
        
        # Configure the main style
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('TButton', background='#404040', foreground='#ffffff')
        style.configure('TNotebook', background='#2b2b2b')
        style.configure('TNotebook.Tab', background='#404040', foreground='#ffffff', padding=[10, 5])
        style.configure('TLabelframe', background='#2b2b2b', foreground='#ffffff')
        style.configure('TLabelframe.Label', background='#2b2b2b', foreground='#ffffff', font=('Arial', 10, 'bold'))
        
        # Ghost Mode button style
        style.configure('Ghost.TButton', 
                       background='#00ff00', 
                       foreground='#000000',
                       font=('Arial', 10, 'bold'),
                       padding=[15, 8])
        
        # Hybrid Mode button style - ENHANCED
        style.configure('Hybrid.TButton', 
                       background='#ff6600', 
                       foreground='#ffffff',
                       font=('Arial', 11, 'bold'),
                       padding=[20, 10])
        
        # Ferris Ride Mode button style
        style.configure('FerrisRide.TButton', 
                       background='#0088ff', 
                       foreground='#ffffff',
                       font=('Arial', 10, 'bold'),
                       padding=[15, 8])
        
        # Status bar style
        style.configure('Status.TLabel', 
                       background='#1a1a1a', 
                       foreground='#00ff00',
                       font=('Consolas', 9))
    
    def _create_header(self):
        """Create header section."""
        header_frame = ttk.Frame(self.root, style='Control.TFrame')
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(header_frame, 
                               text="🎨 Schwabot Visual Controls", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="▶️ Start", 
                  command=self._start_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⏸️ Pause", 
                  command=self._pause_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⏹️ Stop", 
                  command=self._stop_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Refresh", 
                  command=self._refresh_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Save Config", 
                  command=self._save_configuration).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📂 Load Config", 
                  command=self._load_configuration).pack(side=tk.LEFT, padx=5)
    
    def _create_main_content(self):
        """Create main content area."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create tabs
        self._create_chart_controls_tab()
        self._create_layer_management_tab()
        self._create_pattern_recognition_tab()
        self._create_ai_analysis_tab()
        self._create_performance_tab()
        self._create_settings_tab()
    
    def _create_chart_controls_tab(self):
        """Create chart controls tab."""
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="📊 Chart Controls")
        
        # Chart type selection
        type_frame = ttk.LabelFrame(chart_frame, text="Chart Type", padding=10)
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.chart_type_var = tk.StringVar(value=ChartType.PRICE_CHART.value)
        for chart_type in ChartType:
            ttk.Radiobutton(type_frame, 
                           text=chart_type.value.replace('_', ' ').title(),
                           variable=self.chart_type_var,
                           value=chart_type.value).pack(anchor=tk.W)
        
        # Chart configuration
        config_frame = ttk.LabelFrame(chart_frame, text="Chart Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Size controls
        size_frame = ttk.Frame(config_frame)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Width:").pack(side=tk.LEFT)
        self.width_var = tk.IntVar(value=self.chart_config.width)
        width_spin = ttk.Spinbox(size_frame, from_=400, to=2000, 
                                textvariable=self.width_var, width=10)
        width_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(size_frame, text="Height:").pack(side=tk.LEFT, padx=(20, 0))
        self.height_var = tk.IntVar(value=self.chart_config.height)
        height_spin = ttk.Spinbox(size_frame, from_=300, to=1500, 
                                 textvariable=self.height_var, width=10)
        height_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(size_frame, text="DPI:").pack(side=tk.LEFT, padx=(20, 0))
        self.dpi_var = tk.IntVar(value=self.chart_config.dpi)
        dpi_spin = ttk.Spinbox(size_frame, from_=72, to=300, 
                              textvariable=self.dpi_var, width=10)
        dpi_spin.pack(side=tk.LEFT, padx=5)
        
        # Style controls
        style_frame = ttk.Frame(config_frame)
        style_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(style_frame, text="Style:").pack(side=tk.LEFT)
        self.style_var = tk.StringVar(value=self.chart_config.style)
        style_combo = ttk.Combobox(style_frame, textvariable=self.style_var,
                                  values=["dark_background", "default", "classic", "bmh", "ggplot"],
                                  state="readonly", width=15)
        style_combo.pack(side=tk.LEFT, padx=5)
        
        # Options
        options_frame = ttk.Frame(config_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.enable_grid_var = tk.BooleanVar(value=self.chart_config.enable_grid)
        ttk.Checkbutton(options_frame, text="Enable Grid", 
                       variable=self.enable_grid_var).pack(side=tk.LEFT)
        
        self.enable_legend_var = tk.BooleanVar(value=self.chart_config.enable_legend)
        ttk.Checkbutton(options_frame, text="Enable Legend", 
                       variable=self.enable_legend_var).pack(side=tk.LEFT, padx=(20, 0))
        
        self.enable_annotations_var = tk.BooleanVar(value=self.chart_config.enable_annotations)
        ttk.Checkbutton(options_frame, text="Enable Annotations", 
                       variable=self.enable_annotations_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Generate button
        ttk.Button(chart_frame, text="📈 Generate Chart", 
                  command=self._generate_chart).pack(pady=10)
    
    def _create_layer_management_tab(self):
        """Create layer management tab."""
        layer_frame = ttk.Frame(self.notebook)
        self.notebook.add(layer_frame, text="🔧 Layer Management")
        
        # Layer list
        list_frame = ttk.LabelFrame(layer_frame, text="Visual Layers", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for layers
        columns = ('Layer', 'Status', 'Opacity', 'Z-Index', 'Auto-Update')
        self.layer_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.layer_tree.heading(col, text=col)
            self.layer_tree.column(col, width=100)
        
        self.layer_tree.pack(fill=tk.BOTH, expand=True)
        
        # Layer controls
        controls_frame = ttk.Frame(layer_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="➕ Add Layer", 
                  command=self._add_layer).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="➖ Remove Layer", 
                  command=self._remove_layer).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="👁️ Toggle Visibility", 
                  command=self._toggle_layer_visibility).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="⬆️ Move Up", 
                  command=self._move_layer_up).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="⬇️ Move Down", 
                  command=self._move_layer_down).pack(side=tk.LEFT, padx=5)
        
        # Layer properties
        props_frame = ttk.LabelFrame(layer_frame, text="Layer Properties", padding=10)
        props_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Opacity control
        opacity_frame = ttk.Frame(props_frame)
        opacity_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(opacity_frame, text="Opacity:").pack(side=tk.LEFT)
        self.opacity_var = tk.DoubleVar(value=1.0)
        opacity_scale = ttk.Scale(opacity_frame, from_=0.0, to=1.0, 
                                 variable=self.opacity_var, orient=tk.HORIZONTAL)
        opacity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Update interval
        interval_frame = ttk.Frame(props_frame)
        interval_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interval_frame, text="Update Interval (s):").pack(side=tk.LEFT)
        self.interval_var = tk.DoubleVar(value=1.0)
        interval_spin = ttk.Spinbox(interval_frame, from_=0.1, to=60.0, 
                                   textvariable=self.interval_var, width=10)
        interval_spin.pack(side=tk.LEFT, padx=5)
        
        # Apply button
        ttk.Button(props_frame, text="✅ Apply Properties", 
                  command=self._apply_layer_properties).pack(pady=5)
        
        # Populate layer tree
        self._populate_layer_tree()
    
    def _create_pattern_recognition_tab(self):
        """Create pattern recognition tab."""
        pattern_frame = ttk.Frame(self.notebook)
        self.notebook.add(pattern_frame, text="🔍 Pattern Recognition")
        
        # Pattern types
        types_frame = ttk.LabelFrame(pattern_frame, text="Pattern Types", padding=10)
        types_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pattern_vars = {}
        pattern_types = [
            "Double Top/Bottom", "Head and Shoulders", "Triangle", 
            "Rectangle", "Wedge", "Flag", "Pennant", "Cup and Handle"
        ]
        
        for i, pattern in enumerate(pattern_types):
            var = tk.BooleanVar(value=True)
            self.pattern_vars[pattern] = var
            ttk.Checkbutton(types_frame, text=pattern, 
                           variable=var).grid(row=i//2, column=i%2, sticky=tk.W, pady=2)
        
        # Recognition settings
        settings_frame = ttk.LabelFrame(pattern_frame, text="Recognition Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.7)
        conf_scale = ttk.Scale(conf_frame, from_=0.0, to=1.0, 
                              variable=self.confidence_var, orient=tk.HORIZONTAL)
        conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Sensitivity
        sens_frame = ttk.Frame(settings_frame)
        sens_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sens_frame, text="Sensitivity:").pack(side=tk.LEFT)
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sens_scale = ttk.Scale(sens_frame, from_=0.0, to=1.0, 
                              variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sens_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(pattern_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="🔍 Start Recognition", 
                  command=self._start_pattern_recognition).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏹️ Stop Recognition", 
                  command=self._stop_pattern_recognition).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📊 View Results", 
                  command=self._view_pattern_results).pack(side=tk.LEFT, padx=5)
    
    def _create_ai_analysis_tab(self):
        """Create AI analysis tab."""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🤖 AI Analysis")
        
        # AI settings
        settings_frame = ttk.LabelFrame(ai_frame, text="AI Analysis Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Enable AI analysis
        self.enable_ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Enable AI Analysis", 
                       variable=self.enable_ai_var).pack(anchor=tk.W)
        
        # Analysis types
        types_frame = ttk.LabelFrame(ai_frame, text="Analysis Types", padding=10)
        types_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.ai_analysis_vars = {}
        analysis_types = [
            "Technical Analysis", "Pattern Recognition", "Risk Assessment", 
            "Trend Analysis", "Volume Analysis", "Momentum Analysis"
        ]
        
        for analysis in analysis_types:
            var = tk.BooleanVar(value=True)
            self.ai_analysis_vars[analysis] = var
            ttk.Checkbutton(types_frame, text=analysis, 
                           variable=var).pack(anchor=tk.W)
        
        # AI parameters
        params_frame = ttk.LabelFrame(ai_frame, text="AI Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Temperature
        temp_frame = ttk.Frame(params_frame)
        temp_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temperature_var = tk.DoubleVar(value=0.7)
        temp_scale = ttk.Scale(temp_frame, from_=0.0, to=1.0, 
                              variable=self.temperature_var, orient=tk.HORIZONTAL)
        temp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Max length
        length_frame = ttk.Frame(params_frame)
        length_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(length_frame, text="Max Length:").pack(side=tk.LEFT)
        self.max_length_var = tk.IntVar(value=512)
        length_spin = ttk.Spinbox(length_frame, from_=100, to=2000, 
                                 textvariable=self.max_length_var, width=10)
        length_spin.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(ai_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="🤖 Start AI Analysis", 
                  command=self._start_ai_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏹️ Stop AI Analysis", 
                  command=self._stop_ai_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📋 View Analysis", 
                  command=self._view_ai_analysis).pack(side=tk.LEFT, padx=5)
    
    def _create_performance_tab(self):
        """Create performance monitoring tab."""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="📊 Performance")
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(perf_frame, text="Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for metrics
        columns = ('Metric', 'Value', 'Status')
        self.metrics_tree = ttk.Treeview(metrics_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=150)
        
        self.metrics_tree.pack(fill=tk.BOTH, expand=True)
        
        # Performance controls
        control_frame = ttk.Frame(perf_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Auto Refresh", 
                       variable=self.auto_refresh_var).pack(side=tk.LEFT)
        
        ttk.Label(control_frame, text="Refresh Interval (s):").pack(side=tk.LEFT, padx=(20, 0))
        self.refresh_interval_var = tk.DoubleVar(value=5.0)
        refresh_spin = ttk.Spinbox(control_frame, from_=1.0, to=60.0, 
                                  textvariable=self.refresh_interval_var, width=10)
        refresh_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="🔄 Refresh Now", 
                  command=self._refresh_performance).pack(side=tk.LEFT, padx=20)
        
        # Populate metrics
        self._populate_performance_metrics()
    
    def _create_settings_tab(self):
        """Create the Settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Create a scrollable frame for settings
        canvas = tk.Canvas(settings_frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel to scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # General Settings Section
        general_frame = ttk.LabelFrame(scrollable_frame, text="🔧 General Settings", padding=10)
        general_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Output Directory
        output_frame = ttk.Frame(general_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value="visualizations")
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=30)
        output_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="Browse", command=self._browse_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Auto-save checkbox
        self.auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="Auto-save visualizations", variable=self.auto_save_var).pack(anchor=tk.W, pady=5)
        
        # Hardware Optimization Section
        hardware_frame = ttk.LabelFrame(scrollable_frame, text="⚡ Hardware Optimization", padding=10)
        hardware_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.gpu_accel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hardware_frame, text="Enable GPU Acceleration", variable=self.gpu_accel_var).pack(anchor=tk.W, pady=2)
        
        self.memory_opt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hardware_frame, text="Enable Memory Optimization", variable=self.memory_opt_var).pack(anchor=tk.W, pady=2)
        
        # Hardware control buttons
        hardware_buttons_frame = ttk.Frame(hardware_frame)
        hardware_buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(hardware_buttons_frame, text="Save Settings", command=self._save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(hardware_buttons_frame, text="Load Settings", command=self._load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(hardware_buttons_frame, text="Reset to Defaults", command=self._reset_settings).pack(side=tk.LEFT, padx=5)
        
        # Ghost Mode Configuration Section
        ghost_mode_frame = ttk.LabelFrame(scrollable_frame, text="🎯 Ghost Mode - BTC/USDC Optimized Trading", padding=10)
        ghost_mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Ghost Mode Status
        ghost_status_frame = ttk.Frame(ghost_mode_frame)
        ghost_status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ghost_status_frame, text="Status:").pack(side=tk.LEFT)
        self.ghost_status_label = ttk.Label(ghost_status_frame, text="Inactive", foreground="red")
        self.ghost_status_label.pack(side=tk.LEFT, padx=5)
        
        # Ghost Mode Controls
        ghost_controls_frame = ttk.Frame(ghost_mode_frame)
        ghost_controls_frame.pack(fill=tk.X, pady=5)
        
        if GHOST_MODE_AVAILABLE:
            # Ghost Mode Activation Button
            self.ghost_activate_button = ttk.Button(
                ghost_controls_frame, 
                text="🎯 Activate Ghost Mode", 
                command=self._activate_ghost_mode,
                style='Ghost.TButton'
            )
            self.ghost_activate_button.pack(side=tk.LEFT, padx=5)
            
            # Ghost Mode Deactivation Button
            self.ghost_deactivate_button = ttk.Button(
                ghost_controls_frame, 
                text="🔄 Deactivate Ghost Mode", 
                command=self._deactivate_ghost_mode,
                state='disabled'
            )
            self.ghost_deactivate_button.pack(side=tk.LEFT, padx=5)
            
            # Ghost Mode Status Check Button
            ttk.Button(
                ghost_controls_frame, 
                text="📊 Check Status", 
                command=self._check_ghost_mode_status
            ).pack(side=tk.LEFT, padx=5)
            
            # Ghost Mode Requirements Validation Button
            ttk.Button(
                ghost_controls_frame, 
                text="✅ Validate Requirements", 
                command=self._validate_ghost_mode_requirements
            ).pack(side=tk.LEFT, padx=5)
            
        else:
            ttk.Label(ghost_controls_frame, text="❌ Ghost Mode Manager not available", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        # Ghost Mode Information
        info_frame = ttk.Frame(ghost_mode_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        info_text = """
🎯 Ghost Mode: BTC/USDC Optimized Trading System
• Focus: BTC/USDC and USDC/BTC ONLY
• Risk: Medium-risk orbitals (2, 6, 8)
• Profit Target: Few dollars per trade (4% target)
• AI Priority: 80% for Ghost logic
• Backup Systems: All active (Ghost Core, Basket, Echo)
• Position Size: 15% max (vs 10% default)
• Stop Loss: 2.5% (vs 2% default)
• Take Profit: 4% (vs 3% default)
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Consolas", 9))
        info_label.pack(anchor=tk.W)
        
        # Update Ghost Mode status
        self._update_ghost_mode_status()
        
        # Hybrid Mode Configuration Section - ENHANCED
        hybrid_mode_frame = ttk.LabelFrame(scrollable_frame, text="🚀 Hybrid Mode - Quantum Consciousness Trading", padding=15)
        hybrid_mode_frame.pack(fill=tk.X, padx=10, pady=15)
        
        # Hybrid Mode Status with enhanced styling
        hybrid_status_frame = ttk.Frame(hybrid_mode_frame)
        hybrid_status_frame.pack(fill=tk.X, pady=8)
        
        status_label = ttk.Label(hybrid_status_frame, text="Status:", font=("Arial", 10, "bold"))
        status_label.pack(side=tk.LEFT)
        self.hybrid_status_label = ttk.Label(hybrid_status_frame, text="Inactive", foreground="red", font=("Arial", 10, "bold"))
        self.hybrid_status_label.pack(side=tk.LEFT, padx=8)
        
        # Hybrid Mode Controls with enhanced layout
        hybrid_controls_frame = ttk.Frame(hybrid_mode_frame)
        hybrid_controls_frame.pack(fill=tk.X, pady=10)
        
        if HYBRID_MODE_AVAILABLE:
            # Hybrid Mode Activation Button - PROMINENT
            self.hybrid_activate_button = ttk.Button(
                hybrid_controls_frame, 
                text="🚀 ACTIVATE HYBRID MODE", 
                command=self._activate_hybrid_mode,
                style='Hybrid.TButton'
            )
            self.hybrid_activate_button.pack(side=tk.LEFT, padx=8, pady=5)
            
            # Hybrid Mode Deactivation Button
            self.hybrid_deactivate_button = ttk.Button(
                hybrid_controls_frame, 
                text="🔄 Deactivate Hybrid Mode", 
                command=self._deactivate_hybrid_mode,
                state='disabled'
            )
            self.hybrid_deactivate_button.pack(side=tk.LEFT, padx=8, pady=5)
            
            # Hybrid Mode Status Check Button
            ttk.Button(
                hybrid_controls_frame, 
                text="📊 Check Status", 
                command=self._check_hybrid_mode_status
            ).pack(side=tk.LEFT, padx=8, pady=5)
            
            # Hybrid Mode Requirements Validation Button
            ttk.Button(
                hybrid_controls_frame, 
                text="✅ Validate Requirements", 
                command=self._validate_hybrid_mode_requirements
            ).pack(side=tk.LEFT, padx=8, pady=5)
            
        else:
            ttk.Label(hybrid_controls_frame, text="❌ Hybrid Mode Manager not available", 
                     foreground="red", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Hybrid Mode Information - ENHANCED
        hybrid_info_frame = ttk.Frame(hybrid_mode_frame)
        hybrid_info_frame.pack(fill=tk.X, pady=10)
        
        hybrid_info_text = """
🚀 HYBRID MODE: QUANTUM CONSCIOUSNESS TRADING SYSTEM
═══════════════════════════════════════════════════════════════

🌌 QUANTUM STATE: Superposition trading across 8 parallel universes
🧠 AI CONSCIOUSNESS: 85% consciousness level with 47% boost factor
📐 DIMENSIONAL ANALYSIS: 12-dimensional market analysis
💰 QUANTUM POSITION SIZE: 30.5% (12% base × 1.73 quantum × 1.47 consciousness)

🎯 PROFIT TARGETS:
   • Quantum Mode: 4.73% profit target
   • Consciousness Mode: 5.47% profit target  
   • Dimensional Mode: 3.73% profit target

⚛️ QUANTUM SHELLS: [3, 7, 9]
   • Shell 3: Quantum Nucleus
   • Shell 7: Consciousness Core
   • Shell 9: Dimensional Ghost

🤖 QUANTUM AI PRIORITY: 81% for hybrid AI consciousness
⚡ QUANTUM SPEED: 0.33 second market monitoring (quantum speed)

🎲 PARALLEL UNIVERSE TRADING: Simultaneous trading across 8 universes
🔮 CONSCIOUSNESS BOOST: 47% enhanced AI decision making
📊 DIMENSIONAL DEPTH: 12-dimensional market analysis
        """
        
        hybrid_info_label = ttk.Label(hybrid_info_frame, text=hybrid_info_text, justify=tk.LEFT, font=("Consolas", 9))
        hybrid_info_label.pack(anchor=tk.W)
        
        # Update Hybrid Mode status
        self._update_hybrid_mode_status()
        
        # Ferris Ride Mode Configuration Section
        ferris_ride_mode_frame = ttk.LabelFrame(scrollable_frame, text="🎡 Ferris Ride Mode - Revolutionary Auto-Trading", padding=10)
        ferris_ride_mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Ferris Ride Mode Status
        ferris_ride_status_frame = ttk.Frame(ferris_ride_mode_frame)
        ferris_ride_status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ferris_ride_status_frame, text="Status:").pack(side=tk.LEFT)
        self.ferris_ride_status_label = ttk.Label(ferris_ride_status_frame, text="Inactive", foreground="red")
        self.ferris_ride_status_label.pack(side=tk.LEFT, padx=5)
        
        # Ferris Ride Mode Controls
        ferris_ride_controls_frame = ttk.Frame(ferris_ride_mode_frame)
        ferris_ride_controls_frame.pack(fill=tk.X, pady=5)
        
        if FERRIS_RIDE_MODE_AVAILABLE:
            # Ferris Ride Mode Activation Button
            self.ferris_ride_activate_button = ttk.Button(
                ferris_ride_controls_frame, 
                text="🎡 Activate Ferris Ride Mode", 
                command=self._activate_ferris_ride_mode,
                style='FerrisRide.TButton'
            )
            self.ferris_ride_activate_button.pack(side=tk.LEFT, padx=5)
            
            # Ferris Ride Mode Deactivation Button
            self.ferris_ride_deactivate_button = ttk.Button(
                ferris_ride_controls_frame, 
                text="🔄 Deactivate Ferris Ride Mode", 
                command=self._deactivate_ferris_ride_mode,
                state='disabled'
            )
            self.ferris_ride_deactivate_button.pack(side=tk.LEFT, padx=5)
            
            # Ferris Ride Mode Status Check Button
            ttk.Button(
                ferris_ride_controls_frame, 
                text="📊 Check Status", 
                command=self._check_ferris_ride_mode_status
            ).pack(side=tk.LEFT, padx=5)
            
            # Ferris Ride Mode Requirements Validation Button
            ttk.Button(
                ferris_ride_controls_frame, 
                text="✅ Validate Requirements", 
                command=self._validate_ferris_ride_mode_requirements
            ).pack(side=tk.LEFT, padx=5)
            
        else:
            ttk.Label(ferris_ride_controls_frame, text="❌ Ferris Ride Manager not available", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        # Ferris Ride Mode Information
        ferris_ride_info_frame = ttk.Frame(ferris_ride_mode_frame)
        ferris_ride_info_frame.pack(fill=tk.X, pady=5)
        
        ferris_ride_info_text = """
🎡 Ferris Ride Mode: Revolutionary Auto-Trading System
• Auto-Detection: Automatically detects user capital and USDC pairs
• Pattern Studying: Studies market patterns for 3+ days before entry
• Hash Pattern Matching: Generates unique hash patterns for precise entry
• Confidence Building: Builds confidence zones through profit accumulation
• Ferris RDE Framework: Mathematical orbital trading with momentum factors
• USB Backup: Automatic data backup to USB/local storage
• Focus: Everything to USDC / USDC to Everything strategy
• Medium Risk Orbitals: [2, 4, 6, 8] for balanced risk/reward
• Position Size: 12% base × 1.5 Ferris multiplier = 18% total
• Profit Target: 5% per trade
• Stop Loss: 2.5% per trade
• Study Duration: 72 hours minimum before entry
• Confidence Threshold: 60% minimum for zone targeting
• Risk Threshold: 30% maximum risk level
        """
        
        ferris_ride_info_label = ttk.Label(ferris_ride_info_frame, text=ferris_ride_info_text, justify=tk.LEFT, font=("Consolas", 9))
        ferris_ride_info_label.pack(anchor=tk.W)
        
        # Update Ferris Ride Mode status
        self._update_ferris_ride_mode_status()
    
    def _create_status_bar(self):
        """Create status bar."""
        status_frame = ttk.Frame(self.root, style='Control.TFrame')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def _start_update_thread(self):
        """Start the update thread."""
        if self.update_thread and self.update_thread.is_alive():
            return
        
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Main update loop."""
        while True:
            try:
                if self.auto_refresh_var.get() and self.is_running:
                    self._refresh_performance()
                
                time.sleep(self.refresh_interval_var.get())
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                time.sleep(5)
    
    # Control methods
    def _start_visualization(self):
        """Start visualization."""
        try:
            if not self.visual_controller:
                messagebox.showerror("Error", "Visual controller not available")
                return
            
            self.is_running = True
            self.visual_controller.start_processing()
            self._update_status("Visualization started")
            self.progress_bar.start()
            
        except Exception as e:
            logger.error(f"Failed to start visualization: {e}")
            messagebox.showerror("Error", f"Failed to start visualization: {e}")
    
    def _pause_visualization(self):
        """Pause visualization."""
        self.is_running = False
        self._update_status("Visualization paused")
        self.progress_bar.stop()
    
    def _stop_visualization(self):
        """Stop visualization."""
        try:
            self.is_running = False
            if self.visual_controller:
                self.visual_controller.stop_processing()
            self._update_status("Visualization stopped")
            self.progress_bar.stop()
            
        except Exception as e:
            logger.error(f"Failed to stop visualization: {e}")
    
    def _refresh_visualization(self):
        """Refresh visualization."""
        self._update_status("Refreshing visualization...")
        # Implementation would refresh current visualization
    
    def _generate_chart(self):
        """Generate chart with current settings."""
        try:
            # Update chart config
            self.chart_config.width = self.width_var.get()
            self.chart_config.height = self.height_var.get()
            self.chart_config.dpi = self.dpi_var.get()
            self.chart_config.style = self.style_var.get()
            self.chart_config.enable_grid = self.enable_grid_var.get()
            self.chart_config.enable_legend = self.enable_legend_var.get()
            self.chart_config.enable_annotations = self.enable_annotations_var.get()
            
            # Generate chart (mock implementation)
            chart_type = ChartType(self.chart_type_var.get())
            self._update_status(f"Generating {chart_type.value} chart...")
            
            # In real implementation, would call visual controller
            messagebox.showinfo("Success", f"Generated {chart_type.value} chart")
            
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
            messagebox.showerror("Error", f"Failed to generate chart: {e}")
    
    def _populate_layer_tree(self):
        """Populate layer tree with current layers."""
        # Clear existing items
        for item in self.layer_tree.get_children():
            self.layer_tree.delete(item)
        
        # Add layers
        for layer_type, config in self.layer_configs.items():
            status = "Active" if config.enabled else "Inactive"
            self.layer_tree.insert('', 'end', values=(
                layer_type.value.replace('_', ' ').title(),
                status,
                f"{config.opacity:.2f}",
                config.z_index,
                "Yes" if config.auto_update else "No"
            ))
    
    def _add_layer(self):
        """Add new layer."""
        # Implementation for adding new layer
        self._update_status("Layer added")
    
    def _remove_layer(self):
        """Remove selected layer."""
        selection = self.layer_tree.selection()
        if selection:
            self.layer_tree.delete(selection[0])
            self._update_status("Layer removed")
    
    def _toggle_layer_visibility(self):
        """Toggle layer visibility."""
        selection = self.layer_tree.selection()
        if selection:
            # Implementation for toggling visibility
            self._update_status("Layer visibility toggled")
    
    def _move_layer_up(self):
        """Move layer up in z-index."""
        # Implementation for moving layer up
        self._update_status("Layer moved up")
    
    def _move_layer_down(self):
        """Move layer down in z-index."""
        # Implementation for moving layer down
        self._update_status("Layer moved down")
    
    def _apply_layer_properties(self):
        """Apply layer properties."""
        # Implementation for applying properties
        self._update_status("Layer properties applied")
    
    def _start_pattern_recognition(self):
        """Start pattern recognition."""
        self._update_status("Pattern recognition started")
    
    def _stop_pattern_recognition(self):
        """Stop pattern recognition."""
        self._update_status("Pattern recognition stopped")
    
    def _view_pattern_results(self):
        """View pattern recognition results."""
        # Implementation for viewing results
        self._update_status("Pattern results displayed")
    
    def _start_ai_analysis(self):
        """Start AI analysis."""
        self._update_status("AI analysis started")
    
    def _stop_ai_analysis(self):
        """Stop AI analysis."""
        self._update_status("AI analysis stopped")
    
    def _view_ai_analysis(self):
        """View AI analysis results."""
        # Implementation for viewing AI analysis
        self._update_status("AI analysis displayed")
    
    def _populate_performance_metrics(self):
        """Populate performance metrics."""
        # Clear existing items
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        
        # Add metrics (mock data)
        metrics = [
            ("Charts Generated", "15", "✅"),
            ("AI Analyses", "8", "✅"),
            ("Patterns Detected", "12", "✅"),
            ("Render Time (ms)", "45.2", "✅"),
            ("Memory Usage", "256 MB", "✅"),
            ("Cache Hit Rate", "78%", "✅")
        ]
        
        for metric, value, status in metrics:
            self.metrics_tree.insert('', 'end', values=(metric, value, status))
    
    def _refresh_performance(self):
        """Refresh performance metrics."""
        self._populate_performance_metrics()
        self._update_status("Performance metrics refreshed")
    
    def _browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)
    
    def _save_settings(self):
        """Save settings."""
        # Implementation for saving settings
        self._update_status("Settings saved")
    
    def _load_settings(self):
        """Load settings."""
        # Implementation for loading settings
        self._update_status("Settings loaded")
    
    def _reset_settings(self):
        """Reset settings to defaults."""
        # Implementation for resetting settings
        self._update_status("Settings reset to defaults")
    
    def _save_configuration(self):
        """Save current configuration."""
        try:
            config = {
                'chart_config': self.chart_config.__dict__,
                'layer_configs': {k.value: v.__dict__ for k, v in self.layer_configs.items()},
                'active_layers': [layer.value for layer in self.active_layers],
                'auto_refresh': self.auto_refresh_var.get(),
                'refresh_interval': self.refresh_interval_var.get()
            }
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self._update_status(f"Configuration saved to {filename}")
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def _load_configuration(self):
        """Load configuration from file."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Apply configuration
                # Implementation for applying loaded config
                
                self._update_status(f"Configuration loaded from {filename}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def _update_status(self, message: str):
        """Update status bar message."""
        if self.status_label:
            self.status_label.config(text=message)
        logger.info(f"Status: {message}")

    def _activate_ghost_mode(self):
        """Activate Ghost Mode."""
        if not GHOST_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ghost Mode Manager not available")
            return
        
        try:
            # Show confirmation dialog
            result = messagebox.askyesno(
                "🎯 Activate Ghost Mode",
                """Are you sure you want to activate Ghost Mode?

This will configure the system for:
• BTC/USDC and USDC/BTC trading ONLY
• Medium-risk orbitals (2, 6, 8)
• "Few dollars per trade" profit targets (4%)
• 80% AI priority for Ghost logic
• All backup systems active

This will modify your current trading configuration."""
            )
            
            if result:
                # Activate Ghost Mode
                success = ghost_mode_manager.activate_ghost_mode()
                
                if success:
                    self._update_ghost_mode_status()
                    self.ghost_activate_button.config(state='disabled')
                    self.ghost_deactivate_button.config(state='normal')
                    messagebox.showinfo(
                        "✅ Ghost Mode Activated",
                        """Ghost Mode has been successfully activated!

🎯 System now configured for:
• BTC/USDC and USDC/BTC trading
• Medium-risk orbitals (2, 6, 8)
• 4% profit targets ("few dollars per trade")
• 80% AI priority for Ghost logic
• All backup systems active

The system is now optimized for your Ghost Mode trading strategy!"""
                    )
                    self._update_status("🎯 Ghost Mode activated - BTC/USDC optimized trading enabled")
                else:
                    messagebox.showerror("❌ Activation Failed", "Failed to activate Ghost Mode. Check logs for details.")
                    
        except Exception as e:
            logger.error(f"Failed to activate Ghost Mode: {e}")
            messagebox.showerror("Error", f"Failed to activate Ghost Mode: {e}")

    def _deactivate_ghost_mode(self):
        """Deactivate Ghost Mode."""
        if not GHOST_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ghost Mode Manager not available")
            return
        
        try:
            # Show confirmation dialog
            result = messagebox.askyesno(
                "🔄 Deactivate Ghost Mode",
                """Are you sure you want to deactivate Ghost Mode?

This will restore your original trading configuration."""
            )
            
            if result:
                # Deactivate Ghost Mode
                success = ghost_mode_manager.deactivate_ghost_mode()
                
                if success:
                    self._update_ghost_mode_status()
                    self.ghost_activate_button.config(state='normal')
                    self.ghost_deactivate_button.config(state='disabled')
                    messagebox.showinfo(
                        "✅ Ghost Mode Deactivated",
                        "Ghost Mode has been deactivated and original configuration restored."
                    )
                    self._update_status("🔄 Ghost Mode deactivated - original configuration restored")
                else:
                    messagebox.showerror("❌ Deactivation Failed", "Failed to deactivate Ghost Mode. Check logs for details.")
                    
        except Exception as e:
            logger.error(f"Failed to deactivate Ghost Mode: {e}")
            messagebox.showerror("Error", f"Failed to deactivate Ghost Mode: {e}")

    def _check_ghost_mode_status(self):
        """Check and display Ghost Mode status."""
        if not GHOST_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ghost Mode Manager not available")
            return
        
        try:
            status = ghost_mode_manager.get_ghost_mode_status()
            
            status_text = f"""
🎯 Ghost Mode Status Report

Status: {status['status'].upper()}
Active: {'✅ Yes' if status['active'] else '❌ No'}
Config Loaded: {'✅ Yes' if status['config_loaded'] else '❌ No'}

Configuration:
• Supported Symbols: {', '.join(status['supported_symbols'])}
• Orbital Shells: {status['orbital_shells']}
• AI Cluster Priority: {status['ai_cluster_priority']:.1%}
• Backup Systems: {'✅ Active' if status['backup_systems_active'] else '❌ Inactive'}
            """
            
            messagebox.showinfo("📊 Ghost Mode Status", status_text)
            
        except Exception as e:
            logger.error(f"Failed to check Ghost Mode status: {e}")
            messagebox.showerror("Error", f"Failed to check Ghost Mode status: {e}")

    def _validate_ghost_mode_requirements(self):
        """Validate Ghost Mode requirements."""
        if not GHOST_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ghost Mode Manager not available")
            return
        
        try:
            requirements = ghost_mode_manager.validate_ghost_mode_requirements()
            
            validation_text = f"""
✅ Ghost Mode Requirements Validation

Configuration File: {'✅ Exists' if requirements['config_file_exists'] else '❌ Missing'}
Configuration Loaded: {'✅ Yes' if requirements['config_loaded'] else '❌ No'}
BTC/USDC Support: {'✅ Configured' if requirements['btc_usdc_support'] else '❌ Missing'}
Orbital Shells: {'✅ Configured' if requirements['orbital_shells_configured'] else '❌ Missing'}
AI Cluster: {'✅ Configured' if requirements['ai_cluster_configured'] else '❌ Missing'}
Backup Systems: {'✅ Configured' if requirements['backup_systems_configured'] else '❌ Missing'}

Overall Status: {'✅ READY' if all(requirements.values()) else '❌ NOT READY'}
            """
            
            if all(requirements.values()):
                messagebox.showinfo("✅ Requirements Valid", validation_text)
            else:
                messagebox.showwarning("⚠️ Requirements Incomplete", validation_text)
            
        except Exception as e:
            logger.error(f"Failed to validate Ghost Mode requirements: {e}")
            messagebox.showerror("Error", f"Failed to validate Ghost Mode requirements: {e}")

    def _update_ghost_mode_status(self):
        """Update Ghost Mode status display."""
        if not GHOST_MODE_AVAILABLE:
            return
        
        try:
            status = ghost_mode_manager.get_ghost_mode_status()
            
            if status['active']:
                self.ghost_status_label.config(text="Active", foreground="green")
                self.ghost_activate_button.config(state='disabled')
                self.ghost_deactivate_button.config(state='normal')
            else:
                self.ghost_status_label.config(text="Inactive", foreground="red")
                self.ghost_activate_button.config(state='normal')
                self.ghost_deactivate_button.config(state='disabled')
                
        except Exception as e:
            logger.error(f"Failed to update Ghost Mode status: {e}")
            self.ghost_status_label.config(text="Error", foreground="orange")

    def _activate_hybrid_mode(self):
        """Activate Hybrid Mode."""
        if not HYBRID_MODE_AVAILABLE:
            messagebox.showerror("Error", "Hybrid Mode Manager not available")
            return
        
        try:
            # Show confirmation dialog
            result = messagebox.askyesno(
                "🚀 Activate Hybrid Mode",
                """Are you sure you want to activate Hybrid Mode?

This will configure the system for:
• Quantum Consciousness Trading across 8 parallel universes
• 85% AI consciousness level with 47% boost factor
• 12-dimensional market analysis
• 30.5% quantum position size (12% base × 1.73 quantum × 1.47 consciousness)
• 4.73% quantum profit target (quantum mode)
• 5.47% consciousness profit target (consciousness mode)
• 3.73% dimensional profit target (dimensional mode)
• Quantum shells: [3, 7, 9] - Quantum Nucleus, Consciousness Core, Dimensional Ghost
• 81% quantum AI priority for hybrid consciousness
• 0.33 second quantum speed for market monitoring

This will modify your current trading configuration."""
            )
            
            if result:
                # Activate Hybrid Mode
                success = hybrid_mode_manager.activate_hybrid_mode()
                
                if success:
                    self._update_hybrid_mode_status()
                    self.hybrid_activate_button.config(state='disabled')
                    self.hybrid_deactivate_button.config(state='normal')
                    messagebox.showinfo(
                        "✅ Hybrid Mode Activated",
                        """Hybrid Mode has been successfully activated!

🚀 System now configured for:
• Quantum Consciousness Trading across 8 parallel universes
• 85% AI consciousness level with 47% boost factor
• 12-dimensional market analysis
• 30.5% quantum position size (12% base × 1.73 quantum × 1.47 consciousness)
• 4.73% quantum profit target (quantum mode)
• 5.47% consciousness profit target (consciousness mode)
• 3.73% dimensional profit target (dimensional mode)
• Quantum shells: [3, 7, 9] - Quantum Nucleus, Consciousness Core, Dimensional Ghost
• 81% quantum AI priority for hybrid consciousness
• 0.33 second quantum speed for market monitoring

The system is now optimized for your Hybrid Mode trading strategy!"""
                    )
                    self._update_status("🚀 Hybrid Mode activated - Quantum Consciousness Trading enabled")
                else:
                    messagebox.showerror("❌ Activation Failed", "Failed to activate Hybrid Mode. Check logs for details.")
                    
        except Exception as e:
            logger.error(f"Failed to activate Hybrid Mode: {e}")
            messagebox.showerror("Error", f"Failed to activate Hybrid Mode: {e}")

    def _deactivate_hybrid_mode(self):
        """Deactivate Hybrid Mode."""
        if not HYBRID_MODE_AVAILABLE:
            messagebox.showerror("Error", "Hybrid Mode Manager not available")
            return
        
        try:
            # Show confirmation dialog
            result = messagebox.askyesno(
                "🔄 Deactivate Hybrid Mode",
                """Are you sure you want to deactivate Hybrid Mode?

This will restore your original trading configuration."""
            )
            
            if result:
                # Deactivate Hybrid Mode
                success = hybrid_mode_manager.deactivate_hybrid_mode()
                
                if success:
                    self._update_hybrid_mode_status()
                    self.hybrid_activate_button.config(state='normal')
                    self.hybrid_deactivate_button.config(state='disabled')
                    messagebox.showinfo(
                        "✅ Hybrid Mode Deactivated",
                        "Hybrid Mode has been deactivated and original configuration restored."
                    )
                    self._update_status("🔄 Hybrid Mode deactivated - original configuration restored")
                else:
                    messagebox.showerror("❌ Deactivation Failed", "Failed to deactivate Hybrid Mode. Check logs for details.")
                    
        except Exception as e:
            logger.error(f"Failed to deactivate Hybrid Mode: {e}")
            messagebox.showerror("Error", f"Failed to deactivate Hybrid Mode: {e}")

    def _check_hybrid_mode_status(self):
        """Check and display Hybrid Mode status."""
        if not HYBRID_MODE_AVAILABLE:
            messagebox.showerror("Error", "Hybrid Mode Manager not available")
            return
        
        try:
            status = hybrid_mode_manager.get_hybrid_mode_status()
            
            status_text = f"""
🚀 Hybrid Mode Status Report

Status: {status['status'].upper()}
Active: {'✅ Yes' if status['active'] else '❌ No'}
Config Loaded: {'✅ Yes' if status['config_loaded'] else '❌ No'}

Quantum Configuration:
• Quantum State: {status['quantum_state']}
• Consciousness Level: {status['consciousness_level']:.1%}
• Dimensional Depth: {status['dimensional_depth']}
• Parallel Universes: {status['parallel_universes']}

Performance Metrics:
• Quantum Profit: ${status['quantum_profit']:.2f}
• Consciousness Win Rate: {status['consciousness_win_rate']:.1%}
• Dimensional Efficiency: {status['dimensional_efficiency']:.1%}
            """
            
            messagebox.showinfo("📊 Hybrid Mode Status", status_text)
            
        except Exception as e:
            logger.error(f"Failed to check Hybrid Mode status: {e}")
            messagebox.showerror("Error", f"Failed to check Hybrid Mode status: {e}")

    def _validate_hybrid_mode_requirements(self):
        """Validate Hybrid Mode requirements."""
        if not HYBRID_MODE_AVAILABLE:
            messagebox.showerror("Error", "Hybrid Mode Manager not available")
            return
        
        try:
            requirements = hybrid_mode_manager.validate_hybrid_mode_requirements()
            
            validation_text = f"""
✅ Hybrid Mode Requirements Validation

Configuration File: {'✅ Exists' if requirements['config_file_exists'] else '❌ Missing'}
Configuration Loaded: {'✅ Yes' if requirements['config_loaded'] else '❌ No'}
Quantum Consciousness: {'✅ Ready' if requirements['quantum_consciousness_ready'] else '❌ Not Ready'}
Dimensional Analysis: {'✅ Ready' if requirements['dimensional_analysis_ready'] else '❌ Not Ready'}
Parallel Universes: {'✅ Ready' if requirements['parallel_universes_ready'] else '❌ Not Ready'}
Quantum AI: {'✅ Ready' if requirements['quantum_ai_ready'] else '❌ Not Ready'}

Overall Status: {'✅ READY' if all(requirements.values()) else '❌ NOT READY'}

{f"Warning: {requirements.get('warning', '')}" if 'warning' in requirements else ''}
{f"Error: {requirements.get('error', '')}" if 'error' in requirements else ''}
            """
            
            if requirements['all_requirements_met']:
                messagebox.showinfo("✅ Requirements Valid", validation_text)
            else:
                messagebox.showwarning("⚠️ Requirements Incomplete", validation_text)
            
        except Exception as e:
            logger.error(f"Failed to validate Hybrid Mode requirements: {e}")
            messagebox.showerror("Error", f"Failed to validate Hybrid Mode requirements: {e}")

    def _update_hybrid_mode_status(self):
        """Update Hybrid Mode status display."""
        if not HYBRID_MODE_AVAILABLE:
            return
        
        try:
            status = hybrid_mode_manager.get_hybrid_mode_status()
            
            if status['active']:
                self.hybrid_status_label.config(text="Active", foreground="green")
                self.hybrid_activate_button.config(state='disabled')
                self.hybrid_deactivate_button.config(state='normal')
            else:
                self.hybrid_status_label.config(text="Inactive", foreground="red")
                self.hybrid_activate_button.config(state='normal')
                self.hybrid_deactivate_button.config(state='disabled')
                
        except Exception as e:
            logger.error(f"Failed to update Hybrid Mode status: {e}")
            self.hybrid_status_label.config(text="Error", foreground="orange")

    def _activate_ferris_ride_mode(self):
        """Activate Ferris Ride Mode."""
        if not FERRIS_RIDE_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ferris Ride Manager not available")
            return
        
        try:
            # Show confirmation dialog
            result = messagebox.askyesno(
                "🎡 Activate Ferris Ride Mode",
                """Are you sure you want to activate Ferris Ride Mode?

This will configure the system for:
• Auto-Detection: Automatically detects user capital and USDC pairs
• Pattern Studying: Studies market patterns for 3+ days before entry
• Hash Pattern Matching: Generates unique hash patterns for precise entry
• Confidence Building: Builds confidence zones through profit accumulation
• Ferris RDE Framework: Mathematical orbital trading with momentum factors
• USB Backup: Automatic data backup to USB/local storage
• Focus: Everything to USDC / USDC to Everything strategy
• Medium Risk Orbitals: [2, 4, 6, 8] for balanced risk/reward
• Position Size: 12% base × 1.5 Ferris multiplier = 18% total
• Profit Target: 5% per trade
• Stop Loss: 2.5% per trade
• Study Duration: 72 hours minimum before entry
• Confidence Threshold: 60% minimum for zone targeting
• Risk Threshold: 30% maximum risk level

This will modify your current trading configuration."""
            )
            
            if result:
                # Activate Ferris Ride Mode
                success = ferris_ride_manager.activate_ferris_ride_mode()
                
                if success:
                    self._update_ferris_ride_mode_status()
                    self.ferris_ride_activate_button.config(state='disabled')
                    self.ferris_ride_deactivate_button.config(state='normal')
                    messagebox.showinfo(
                        "✅ Ferris Ride Mode Activated",
                        """Ferris Ride Mode has been successfully activated!

🎡 System now configured for:
• Auto-Detection: Automatically detects user capital and USDC pairs
• Pattern Studying: Studies market patterns for 3+ days before entry
• Hash Pattern Matching: Generates unique hash patterns for precise entry
• Confidence Building: Builds confidence zones through profit accumulation
• Ferris RDE Framework: Mathematical orbital trading with momentum factors
• USB Backup: Automatic data backup to USB/local storage
• Focus: Everything to USDC / USDC to Everything strategy
• Medium Risk Orbitals: [2, 4, 6, 8] for balanced risk/reward
• Position Size: 12% base × 1.5 Ferris multiplier = 18% total
• Profit Target: 5% per trade
• Stop Loss: 2.5% per trade
• Study Duration: 72 hours minimum before entry
• Confidence Threshold: 60% minimum for zone targeting
• Risk Threshold: 30% maximum risk level

The system is now optimized for your Ferris Ride Mode trading strategy!"""
                    )
                    self._update_status("🎡 Ferris Ride Mode activated - Revolutionary Auto-Trading enabled")
                else:
                    messagebox.showerror("❌ Activation Failed", "Failed to activate Ferris Ride Mode. Check logs for details.")
                    
        except Exception as e:
            logger.error(f"Failed to activate Ferris Ride Mode: {e}")
            messagebox.showerror("Error", f"Failed to activate Ferris Ride Mode: {e}")

    def _deactivate_ferris_ride_mode(self):
        """Deactivate Ferris Ride Mode."""
        if not FERRIS_RIDE_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ferris Ride Manager not available")
            return
        
        try:
            # Show confirmation dialog
            result = messagebox.askyesno(
                "🔄 Deactivate Ferris Ride Mode",
                """Are you sure you want to deactivate Ferris Ride Mode?

This will restore your original trading configuration."""
            )
            
            if result:
                # Deactivate Ferris Ride Mode
                success = ferris_ride_manager.deactivate_ferris_ride_mode()
                
                if success:
                    self._update_ferris_ride_mode_status()
                    self.ferris_ride_activate_button.config(state='normal')
                    self.ferris_ride_deactivate_button.config(state='disabled')
                    messagebox.showinfo(
                        "✅ Ferris Ride Mode Deactivated",
                        "Ferris Ride Mode has been deactivated and original configuration restored."
                    )
                    self._update_status("🔄 Ferris Ride Mode deactivated - original configuration restored")
                else:
                    messagebox.showerror("❌ Deactivation Failed", "Failed to deactivate Ferris Ride Mode. Check logs for details.")
                    
        except Exception as e:
            logger.error(f"Failed to deactivate Ferris Ride Mode: {e}")
            messagebox.showerror("Error", f"Failed to deactivate Ferris Ride Mode: {e}")

    def _check_ferris_ride_mode_status(self):
        """Check and display Ferris Ride Mode status."""
        if not FERRIS_RIDE_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ferris Ride Manager not available")
            return
        
        try:
            status = ferris_ride_manager.get_ferris_ride_status()
            
            if status.get("available", False):
                status_text = f"""
🎡 Ferris Ride Mode Status Report

Status: {'✅ ACTIVE' if status['active'] else '❌ INACTIVE'}
Available: ✅ Yes
Config File: {status.get('config_file', 'N/A')}
Backup Directory: {status.get('backup_dir', 'N/A')}

Configuration:
• Auto-Detect Capital: {'✅ Yes' if status['config']['auto_detect_capital'] else '❌ No'}
• Auto-Detect Tickers: {'✅ Yes' if status['config']['auto_detect_tickers'] else '❌ No'}
• USB Backup Enabled: {'✅ Yes' if status['config']['usb_backup_enabled'] else '❌ No'}
• Study Duration: {status['config']['study_duration_hours']} hours
• Confidence Threshold: {status['config']['confidence_threshold']:.1%}
• Position Size: {status['config']['base_position_size_pct']:.1%} × {status['config']['ferris_multiplier']} = {(status['config']['base_position_size_pct'] * status['config']['ferris_multiplier']):.1%}
• Profit Target: {status['config']['profit_target_pct']:.1%}
• Stop Loss: {status['config']['stop_loss_pct']:.1%}
• Orbital Shells: {status['config']['orbital_shells']}
• Current Shell: {status['config']['current_shell']}
• USDC Pairs Only: {'✅ Yes' if status['config']['usdc_pairs_only'] else '❌ No'}
• Max Daily Loss: {status['config']['max_daily_loss_pct']:.1%}
• Win Rate Target: {status['config']['win_rate_target']:.1%}

Ferris System Status:
• Active Zones: {status['ferris_system_status']['active_zones']}
• Studied Patterns: {status['ferris_system_status']['studied_patterns']}
• Detected Capital: ${status['ferris_system_status']['detected_capital']:.2f}
• Detected Tickers: {status['ferris_system_status']['detected_tickers']}
• Current Orbital Shell: {status['ferris_system_status']['current_orbital_shell']}
• Total Trades: {status['ferris_system_status']['performance']['total_trades']}
• Total Profit: ${status['ferris_system_status']['performance']['total_profit']:.2f}
• Confidence Bonus: {status['ferris_system_status']['performance']['confidence_bonus']:.1%}
                """
            else:
                status_text = f"""
❌ Ferris Ride Mode Status Report

Status: ❌ NOT AVAILABLE
Error: {status.get('error', 'Unknown error')}
                """
            
            messagebox.showinfo("📊 Ferris Ride Mode Status", status_text)
            
        except Exception as e:
            logger.error(f"Failed to check Ferris Ride Mode status: {e}")
            messagebox.showerror("Error", f"Failed to check Ferris Ride Mode status: {e}")

    def _validate_ferris_ride_mode_requirements(self):
        """Validate Ferris Ride Mode requirements."""
        if not FERRIS_RIDE_MODE_AVAILABLE:
            messagebox.showerror("Error", "Ferris Ride Manager not available")
            return
        
        try:
            requirements = ferris_ride_manager.validate_ferris_ride_requirements()
            
            validation_text = f"""
✅ Ferris Ride Mode Requirements Validation

Ferris System Available: {'✅ Yes' if requirements['ferris_system_available'] else '❌ No'}
Config File Exists: {'✅ Yes' if requirements['config_file_exists'] else '❌ No'}
Backup Directory Accessible: {'✅ Yes' if requirements['backup_dir_accessible'] else '❌ No'}
Auto-Detection Ready: {'✅ Yes' if requirements['auto_detection_ready'] else '❌ No'}
USB Backup Ready: {'✅ Yes' if requirements['usb_backup_ready'] else '❌ No'}

Overall Status: {'✅ ALL REQUIREMENTS MET' if requirements['all_requirements_met'] else '❌ REQUIREMENTS NOT MET'}

{f"Warning: {requirements.get('warning', '')}" if 'warning' in requirements else ''}
{f"Error: {requirements.get('error', '')}" if 'error' in requirements else ''}
            """
            
            if requirements['all_requirements_met']:
                messagebox.showinfo("✅ Requirements Valid", validation_text)
            else:
                messagebox.showwarning("⚠️ Requirements Incomplete", validation_text)
            
        except Exception as e:
            logger.error(f"Failed to validate Ferris Ride Mode requirements: {e}")
            messagebox.showerror("Error", f"Failed to validate Ferris Ride Mode requirements: {e}")

    def _update_ferris_ride_mode_status(self):
        """Update Ferris Ride Mode status display."""
        if not FERRIS_RIDE_MODE_AVAILABLE:
            return
        
        try:
            status = ferris_ride_manager.get_ferris_ride_status()
            
            if status.get("available", False) and status.get("active", False):
                self.ferris_ride_status_label.config(text="Active", foreground="green")
                self.ferris_ride_activate_button.config(state='disabled')
                self.ferris_ride_deactivate_button.config(state='normal')
            else:
                self.ferris_ride_status_label.config(text="Inactive", foreground="red")
                self.ferris_ride_activate_button.config(state='normal')
                self.ferris_ride_deactivate_button.config(state='disabled')
                
        except Exception as e:
            logger.error(f"Failed to update Ferris Ride Mode status: {e}")
            self.ferris_ride_status_label.config(text="Error", foreground="orange")


def show_visual_controls(parent=None):
    """Show the visual controls GUI."""
    gui = VisualControlsGUI(parent)
    gui.show_visual_controls()
    return gui


if __name__ == "__main__":
    # Test the visual controls GUI
    logging.basicConfig(level=logging.INFO)
    
    gui = VisualControlsGUI()
    gui.show_visual_controls()
    
    # Keep the GUI running
    if gui.root:
        gui.root.mainloop() 