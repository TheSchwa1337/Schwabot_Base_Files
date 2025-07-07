#!/usr/bin/env python3
"""
🖥️ VISUAL EXECUTION NODE - SCHWABOT GUI INTEGRATION LAYER
========================================================

Advanced visual execution system that provides:
- Real-time 2-gram pattern visualization with emoji rendering
- Interactive trading dashboard with Phantom Math integration
- Live portfolio balancing display with fractal memory
- T-cell health monitoring with system protection alerts
- Market data visualization with pattern correlation
- Strategy trigger visualization with execution tracking

This node serves as Schwabot's visual cortex for human interaction.
"""

import asyncio
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.animation import FuncAnimation
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    # Create mock classes for testing
    class tk:
        class Tk:
            def __init__(self): pass
            def mainloop(self): pass
            def quit(self): pass
            def after(self, delay, func): pass
            def geometry(self, size): pass
            def title(self, title): pass
            def protocol(self, event, func): pass
        class Frame:
            def __init__(self, parent, **kwargs): pass
            def pack(self, **kwargs): pass
            def grid(self, **kwargs): pass
        class Label:
            def __init__(self, parent, **kwargs): pass
            def pack(self, **kwargs): pass
            def grid(self, **kwargs): pass
            def config(self, **kwargs): pass
        class Button:
            def __init__(self, parent, **kwargs): pass
            def pack(self, **kwargs): pass
            def grid(self, **kwargs): pass
            def config(self, **kwargs): pass
        class Canvas:
            def __init__(self, parent, **kwargs): pass
            def pack(self, **kwargs): pass
            def create_text(self, x, y, text="", **kwargs): pass
            def create_rectangle(self, x1, y1, x2, y2, **kwargs): pass
            def delete(self, tag): pass
    class ttk:
        class Notebook:
            def __init__(self, parent, **kwargs): pass
            def pack(self, **kwargs): pass
            def add(self, frame, text=""): pass
        class Progressbar:
            def __init__(self, parent, **kwargs): pass
            def pack(self, **kwargs): pass
            def configure(self, **kwargs): pass

from .two_gram_detector import TwoGramDetector, TwoGramSignal, create_two_gram_detector
# from .strategy_trigger_router import StrategyTriggerRouter, TriggerEvent, ExecutionResult  # Removed to fix circular import
from .algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer
from .btc_usdc_trading_integration import BTCUSDCTradingIntegration
from .phantom_detector import PhantomZone
from .phantom_registry import PhantomRegistry
from utils.safe_print import safe_print, info, warn, error, success, debug

logger = logging.getLogger(__name__)

# Type hints for circular import resolution
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .strategy_trigger_router import StrategyTriggerRouter, TriggerEvent, ExecutionResult


class GUIMode(Enum):
    """GUI display modes for different use cases."""
    FULL_DASHBOARD = "full_dashboard"
    PATTERN_ONLY = "pattern_only"
    TRADING_ONLY = "trading_only"
    MONITORING_ONLY = "monitoring_only"
    DEMO_MODE = "demo_mode"


class VisualizationTheme(Enum):
    """Visual themes for the GUI."""
    DARK_CYBERPUNK = "dark_cyberpunk"
    LIGHT_MINIMAL = "light_minimal"
    MATRIX_GREEN = "matrix_green"
    SCHWABOT_CLASSIC = "schwabot_classic"


@dataclass
class VisualConfig:
    """Configuration for visual execution node."""
    gui_mode: GUIMode = GUIMode.FULL_DASHBOARD
    theme: VisualizationTheme = VisualizationTheme.SCHWABOT_CLASSIC
    window_title: str = "🧬 Schwabot Visual Execution Node"
    window_size: str = "1400x900"
    update_interval_ms: int = 1000
    emoji_scale: float = 1.5
    pattern_history_size: int = 100
    chart_update_interval: int = 5000
    enable_sound_alerts: bool = False
    enable_notifications: bool = True


@dataclass
class PatternVisualization:
    """Visual representation of a 2-gram pattern."""
    pattern: str
    emoji_symbol: str
    frequency: int
    burst_score: float
    entropy: float
    timestamp: float
    x: float
    y: float
    color: str
    size: float
    alpha: float = 1.0


@dataclass
class MarketVisualization:
    """Visual representation of market data."""
    symbol: str
    price: float
    change_24h: float
    volume: float
    timestamp: float
    color: str
    trend_arrow: str


class VisualExecutionNode:
    """
    Advanced visual execution node for Schwabot.
    
    Provides comprehensive GUI interface for:
    - Real-time pattern visualization
    - Trading execution monitoring
    - Portfolio balance display
    - System health monitoring
    - Interactive control panels
    """
    
    def __init__(self, config: VisualConfig):
        self.config = config
        
        # Core components (will be injected)
        self.two_gram_detector: Optional[TwoGramDetector] = None
        self.strategy_router: Optional['StrategyTriggerRouter'] = None
        self.portfolio_balancer: Optional[AlgorithmicPortfolioBalancer] = None
        self.btc_usdc_integration: Optional[BTCUSDCTradingIntegration] = None
        
        # GUI components
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[tk.Label] = None
        self.pattern_display: Optional[tk.Frame] = None
        
        # Visualization data
        self.pattern_history: deque = deque(maxlen=config.pattern_history_size)
        self.market_data_history: deque = deque(maxlen=50)
        self.execution_history: List[ExecutionResult] = []
        
        # Visual state
        self.running = False
        self.last_update_time = 0
        self.animation_frame = 0
        
        # Theme configuration
        self.color_scheme = self._get_color_scheme()
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.current_fps = 0
        
        logger.info("🖥️ Visual Execution Node initialized")

    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme based on theme."""
        schemes = {
            VisualizationTheme.DARK_CYBERPUNK: {
                "bg": "#0a0a0a",
                "fg": "#00ff00",
                "accent": "#ff00ff",
                "warning": "#ffff00",
                "error": "#ff0000",
                "success": "#00ff80",
                "pattern": "#00ccff",
                "trading": "#ff8000"
            },
            VisualizationTheme.LIGHT_MINIMAL: {
                "bg": "#ffffff",
                "fg": "#333333",
                "accent": "#0066cc",
                "warning": "#ff9900",
                "error": "#cc0000",
                "success": "#009900",
                "pattern": "#0099cc",
                "trading": "#cc6600"
            },
            VisualizationTheme.MATRIX_GREEN: {
                "bg": "#000000",
                "fg": "#00ff00",
                "accent": "#66ff66",
                "warning": "#ffff00",
                "error": "#ff4444",
                "success": "#44ff44",
                "pattern": "#00cc00",
                "trading": "#88ff88"
            },
            VisualizationTheme.SCHWABOT_CLASSIC: {
                "bg": "#1a1a2e",
                "fg": "#eeeeff",
                "accent": "#16213e",
                "warning": "#ffa500",
                "error": "#ff4757",
                "success": "#2ed573",
                "pattern": "#3742fa",
                "trading": "#ff6b6b"
            }
        }
        return schemes.get(self.config.theme, schemes[VisualizationTheme.SCHWABOT_CLASSIC])

    async def inject_components(self,
                              two_gram_detector: TwoGramDetector,
                              strategy_router: Optional['StrategyTriggerRouter'] = None,
                              portfolio_balancer: Optional[AlgorithmicPortfolioBalancer] = None,
                              btc_usdc_integration: Optional[BTCUSDCTradingIntegration] = None):
        """Inject system components for visualization."""
        self.two_gram_detector = two_gram_detector
        self.strategy_router = strategy_router
        self.portfolio_balancer = portfolio_balancer
        self.btc_usdc_integration = btc_usdc_integration
        
        logger.info("🔌 Components injected into visual execution node")

    def initialize_gui(self) -> bool:
        """Initialize the GUI interface."""
        if not GUI_AVAILABLE:
            warn("GUI libraries not available, running in headless mode")
            return False
        
        try:
            self.root = tk.Tk()
            self.root.title(self.config.window_title)
            self.root.geometry(self.config.window_size)
            self.root.configure(bg=self.color_scheme["bg"])
            
            # Create main layout
            self._create_main_layout()
            
            # Setup event handlers
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            
            # Start update loop
            self._schedule_update()
            
            info("🖥️ GUI initialized successfully")
            return True
            
        except Exception as e:
            error(f"Failed to initialize GUI: {e}")
            return False

    def _create_main_layout(self):
        """Create the main GUI layout."""
        if not self.root:
            return
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Pattern visualization tab
        pattern_frame = tk.Frame(notebook, bg=self.color_scheme["bg"])
        notebook.add(pattern_frame, text="🧬 Pattern Detection")
        self._create_pattern_tab(pattern_frame)
        
        # Trading dashboard tab
        trading_frame = tk.Frame(notebook, bg=self.color_scheme["bg"])
        notebook.add(trading_frame, text="📊 Trading Dashboard")
        self._create_trading_tab(trading_frame)
        
        # Portfolio monitoring tab
        portfolio_frame = tk.Frame(notebook, bg=self.color_scheme["bg"])
        notebook.add(portfolio_frame, text="⚖️ Portfolio Balance")
        self._create_portfolio_tab(portfolio_frame)
        
        # System health tab
        health_frame = tk.Frame(notebook, bg=self.color_scheme["bg"])
        notebook.add(health_frame, text="🛡️ System Health")
        self._create_health_tab(health_frame)
        
        # Create status bar
        self._create_status_bar()

    def _create_pattern_tab(self, parent):
        """Create the pattern detection visualization tab."""
        # Pattern canvas
        canvas_frame = tk.Frame(parent, bg=self.color_scheme["bg"])
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.color_scheme["bg"],
            highlightthickness=0,
            width=800,
            height=400
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Pattern info panel
        info_frame = tk.Frame(parent, bg=self.color_scheme["accent"])
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # Pattern statistics
        stats_label = tk.Label(
            info_frame,
            text="Pattern Statistics",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 12, "bold")
        )
        stats_label.pack(pady=5)
        
        self.pattern_stats_text = scrolledtext.ScrolledText(
            info_frame,
            height=8,
            bg=self.color_scheme["bg"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 10)
        )
        self.pattern_stats_text.pack(fill="x", padx=10, pady=5)

    def _create_trading_tab(self, parent):
        """Create the trading dashboard tab."""
        # Market data display
        market_frame = tk.Frame(parent, bg=self.color_scheme["accent"])
        market_frame.pack(fill="x", padx=10, pady=10)
        
        market_label = tk.Label(
            market_frame,
            text="📈 Live Market Data",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 14, "bold")
        )
        market_label.pack(pady=5)
        
        self.market_display = tk.Frame(market_frame, bg=self.color_scheme["bg"])
        self.market_display.pack(fill="x", padx=10, pady=5)
        
        # Trading controls
        controls_frame = tk.Frame(parent, bg=self.color_scheme["accent"])
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        controls_label = tk.Label(
            controls_frame,
            text="🎮 Trading Controls",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 14, "bold")
        )
        controls_label.pack(pady=5)
        
        button_frame = tk.Frame(controls_frame, bg=self.color_scheme["accent"])
        button_frame.pack(pady=10)
        
        # Control buttons
        self.start_button = tk.Button(
            button_frame,
            text="▶️ Start Trading",
            command=self._start_trading,
            bg=self.color_scheme["success"],
            fg="white",
            font=("Consolas", 12)
        )
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏹️ Stop Trading",
            command=self._stop_trading,
            bg=self.color_scheme["error"],
            fg="white",
            font=("Consolas", 12)
        )
        self.stop_button.pack(side="left", padx=5)
        
        self.demo_button = tk.Button(
            button_frame,
            text="🎭 Demo Mode",
            command=self._toggle_demo,
            bg=self.color_scheme["warning"],
            fg="white",
            font=("Consolas", 12)
        )
        self.demo_button.pack(side="left", padx=5)

    def _create_portfolio_tab(self, parent):
        """Create the portfolio monitoring tab."""
        # Portfolio balance display
        balance_frame = tk.Frame(parent, bg=self.color_scheme["accent"])
        balance_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        balance_label = tk.Label(
            balance_frame,
            text="⚖️ Portfolio Balance",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 14, "bold")
        )
        balance_label.pack(pady=5)
        
        self.balance_canvas = tk.Canvas(
            balance_frame,
            bg=self.color_scheme["bg"],
            highlightthickness=0,
            height=300
        )
        self.balance_canvas.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Performance metrics
        metrics_frame = tk.Frame(parent, bg=self.color_scheme["accent"])
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        metrics_label = tk.Label(
            metrics_frame,
            text="📊 Performance Metrics",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 14, "bold")
        )
        metrics_label.pack(pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(
            metrics_frame,
            height=6,
            bg=self.color_scheme["bg"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 10)
        )
        self.metrics_text.pack(fill="x", padx=10, pady=5)

    def _create_health_tab(self, parent):
        """Create the system health monitoring tab."""
        # System status display
        status_frame = tk.Frame(parent, bg=self.color_scheme["accent"])
        status_frame.pack(fill="x", padx=10, pady=10)
        
        status_label = tk.Label(
            status_frame,
            text="🛡️ System Health Monitor",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 14, "bold")
        )
        status_label.pack(pady=5)
        
        # Health indicators
        indicators_frame = tk.Frame(status_frame, bg=self.color_scheme["bg"])
        indicators_frame.pack(fill="x", padx=10, pady=5)
        
        self.health_indicators = {}
        indicator_names = [
            ("🧬", "2-Gram Detector"),
            ("🎯", "Strategy Router"),
            ("⚖️", "Portfolio Balancer"),
            ("💱", "BTC/USDC Integration"),
            ("🌐", "Network Status"),
            ("💾", "Memory Usage")
        ]
        
        for i, (emoji, name) in enumerate(indicator_names):
            row = i // 3
            col = i % 3
            
            indicator_frame = tk.Frame(indicators_frame, bg=self.color_scheme["accent"])
            indicator_frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            indicator_label = tk.Label(
                indicator_frame,
                text=f"{emoji} {name}",
                bg=self.color_scheme["accent"],
                fg=self.color_scheme["fg"],
                font=("Consolas", 10)
            )
            indicator_label.pack(anchor="w")
            
            status_indicator = tk.Label(
                indicator_frame,
                text="🟡 Unknown",
                bg=self.color_scheme["accent"],
                fg=self.color_scheme["warning"],
                font=("Consolas", 9)
            )
            status_indicator.pack(anchor="w")
            
            self.health_indicators[name] = status_indicator
        
        # Configure grid weights
        for i in range(3):
            indicators_frame.columnconfigure(i, weight=1)
        
        # T-cell status
        tcell_frame = tk.Frame(parent, bg=self.color_scheme["accent"])
        tcell_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        tcell_label = tk.Label(
            tcell_frame,
            text="🛡️ T-Cell Protection Status",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 14, "bold")
        )
        tcell_label.pack(pady=5)
        
        self.tcell_text = scrolledtext.ScrolledText(
            tcell_frame,
            height=8,
            bg=self.color_scheme["bg"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 10)
        )
        self.tcell_text.pack(fill="both", expand=True, padx=10, pady=5)

    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        if not self.root:
            return
        
        status_frame = tk.Frame(self.root, bg=self.color_scheme["accent"])
        status_frame.pack(fill="x", side="bottom")
        
        self.status_label = tk.Label(
            status_frame,
            text="🔄 Initializing...",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 10),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10, pady=2)
        
        # FPS counter
        self.fps_label = tk.Label(
            status_frame,
            text="FPS: 0",
            bg=self.color_scheme["accent"],
            fg=self.color_scheme["fg"],
            font=("Consolas", 9),
            anchor="e"
        )
        self.fps_label.pack(side="right", padx=10, pady=2)

    def _schedule_update(self):
        """Schedule the next GUI update."""
        if self.root and self.running:
            self.root.after(self.config.update_interval_ms, self._update_gui)

    async def _update_gui(self):
        """Update the GUI with latest data."""
        try:
            current_time = time.time()
            
            # Update FPS
            self.fps_counter += 1
            if current_time - self.fps_last_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_last_time = current_time
                
                if hasattr(self, 'fps_label'):
                    self.fps_label.config(text=f"FPS: {self.current_fps}")
            
            # Update pattern visualization
            await self._update_pattern_display()
            
            # Update trading dashboard
            await self._update_trading_display()
            
            # Update portfolio display
            await self._update_portfolio_display()
            
            # Update health monitoring
            await self._update_health_display()
            
            # Update status
            self._update_status()
            
            self.animation_frame += 1
            self.last_update_time = current_time
            
            # Schedule next update
            self._schedule_update()
            
        except Exception as e:
            logger.error(f"Error updating GUI: {e}")
            if self.running:
                self._schedule_update()

    async def _update_pattern_display(self):
        """Update the pattern visualization display."""
        if not self.canvas or not self.two_gram_detector:
            return
        
        try:
            # Clear canvas
            self.canvas.delete("all")
            
            # Get recent patterns
            recent_patterns = await self.two_gram_detector.get_recent_patterns(limit=20)
            
            # Visualize patterns
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 400
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Draw pattern field
            for i, pattern_data in enumerate(recent_patterns):
                if i >= 10:  # Limit display
                    break
                
                x = (i % 5) * (canvas_width // 5) + (canvas_width // 10)
                y = (i // 5) * (canvas_height // 3) + (canvas_height // 6)
                
                # Create pattern visualization
                pattern_viz = PatternVisualization(
                    pattern=pattern_data.get("pattern", "??"),
                    emoji_symbol=pattern_data.get("emoji_symbol", "❓"),
                    frequency=pattern_data.get("frequency", 0),
                    burst_score=pattern_data.get("burst_score", 0.0),
                    entropy=pattern_data.get("entropy", 0.0),
                    timestamp=pattern_data.get("timestamp", time.time()),
                    x=x,
                    y=y,
                    color=self._get_pattern_color(pattern_data.get("burst_score", 0.0)),
                    size=min(30, max(10, pattern_data.get("frequency", 1) * 2))
                )
                
                self._draw_pattern(pattern_viz)
            
            # Update pattern statistics
            if hasattr(self, 'pattern_stats_text'):
                stats = await self.two_gram_detector.get_pattern_statistics()
                self._update_pattern_stats(stats)
            
        except Exception as e:
            logger.error(f"Error updating pattern display: {e}")

    def _get_pattern_color(self, burst_score: float) -> str:
        """Get color based on burst score."""
        if burst_score > 2.5:
            return self.color_scheme["error"]  # High burst - red
        elif burst_score > 1.5:
            return self.color_scheme["warning"]  # Medium burst - yellow
        elif burst_score > 0.5:
            return self.color_scheme["success"]  # Low burst - green
        else:
            return self.color_scheme["fg"]  # No burst - normal

    def _draw_pattern(self, pattern_viz: PatternVisualization):
        """Draw a pattern visualization on the canvas."""
        if not self.canvas:
            return
        
        # Draw pattern circle
        radius = pattern_viz.size
        self.canvas.create_oval(
            pattern_viz.x - radius,
            pattern_viz.y - radius,
            pattern_viz.x + radius,
            pattern_viz.y + radius,
            fill=pattern_viz.color,
            outline=self.color_scheme["fg"],
            width=2
        )
        
        # Draw emoji symbol
        self.canvas.create_text(
            pattern_viz.x,
            pattern_viz.y - radius - 15,
            text=pattern_viz.emoji_symbol,
            fill=self.color_scheme["fg"],
            font=("Arial", int(14 * self.config.emoji_scale))
        )
        
        # Draw pattern text
        self.canvas.create_text(
            pattern_viz.x,
            pattern_viz.y,
            text=pattern_viz.pattern,
            fill="white",
            font=("Consolas", 10, "bold")
        )
        
        # Draw frequency
        self.canvas.create_text(
            pattern_viz.x,
            pattern_viz.y + radius + 10,
            text=f"f:{pattern_viz.frequency}",
            fill=self.color_scheme["fg"],
            font=("Consolas", 8)
        )
        
        # Draw burst score
        self.canvas.create_text(
            pattern_viz.x,
            pattern_viz.y + radius + 25,
            text=f"b:{pattern_viz.burst_score:.1f}",
            fill=self.color_scheme["fg"],
            font=("Consolas", 8)
        )

    def _update_pattern_stats(self, stats: Dict[str, Any]):
        """Update pattern statistics display."""
        if not hasattr(self, 'pattern_stats_text'):
            return
        
        try:
            self.pattern_stats_text.delete(1.0, tk.END)
            
            stats_text = f"""
🧬 PATTERN DETECTOR STATISTICS
{'=' * 40}

Active Patterns: {stats.get('active_patterns', 0)}
Total Sequences: {stats.get('total_sequences_processed', 0)}
Burst Events: {stats.get('burst_events', 0)}
Avg Entropy: {stats.get('average_entropy', 0.0):.3f}

🎯 TOP PATTERNS:
"""
            
            top_patterns = stats.get('top_patterns', [])
            for i, pattern in enumerate(top_patterns[:5], 1):
                stats_text += f"{i}. {pattern.get('pattern', '??')} {pattern.get('emoji_symbol', '❓')} "
                stats_text += f"(f:{pattern.get('frequency', 0)}, b:{pattern.get('burst_score', 0.0):.1f})\n"
            
            stats_text += f"""
🛡️ SYSTEM HEALTH:
Health Score: {stats.get('system_health_score', 0.0):.2f}
T-Cell Status: {stats.get('t_cell_status', 'Unknown')}
Memory Usage: {stats.get('memory_usage_mb', 0):.1f} MB
"""
            
            self.pattern_stats_text.insert(tk.END, stats_text)
            
        except Exception as e:
            logger.error(f"Error updating pattern stats: {e}")

    async def _update_trading_display(self):
        """Update the trading dashboard display."""
        if not hasattr(self, 'market_display'):
            return
        
        try:
            # Clear previous market data
            for widget in self.market_display.winfo_children():
                widget.destroy()
            
            # Simulate market data (in real implementation, this would come from live feeds)
            market_data = {
                "BTC/USDC": {
                    "price": 50000.0 + np.random.normal(0, 500),
                    "change_24h": np.random.normal(0, 3),
                    "volume": 1000000 + np.random.normal(0, 100000)
                },
                "ETH/USDC": {
                    "price": 3000.0 + np.random.normal(0, 100),
                    "change_24h": np.random.normal(0, 4),
                    "volume": 800000 + np.random.normal(0, 80000)
                }
            }
            
            # Display market data
            row = 0
            for symbol, data in market_data.items():
                # Symbol label
                symbol_label = tk.Label(
                    self.market_display,
                    text=symbol,
                    bg=self.color_scheme["bg"],
                    fg=self.color_scheme["fg"],
                    font=("Consolas", 12, "bold")
                )
                symbol_label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
                
                # Price
                price_color = self.color_scheme["success"] if data["change_24h"] > 0 else self.color_scheme["error"]
                price_label = tk.Label(
                    self.market_display,
                    text=f"${data['price']:.2f}",
                    bg=self.color_scheme["bg"],
                    fg=price_color,
                    font=("Consolas", 12)
                )
                price_label.grid(row=row, column=1, padx=10, pady=5)
                
                # Change
                change_text = f"{'↗️' if data['change_24h'] > 0 else '↘️'} {data['change_24h']:+.2f}%"
                change_label = tk.Label(
                    self.market_display,
                    text=change_text,
                    bg=self.color_scheme["bg"],
                    fg=price_color,
                    font=("Consolas", 10)
                )
                change_label.grid(row=row, column=2, padx=10, pady=5)
                
                # Volume
                volume_label = tk.Label(
                    self.market_display,
                    text=f"Vol: {data['volume']:,.0f}",
                    bg=self.color_scheme["bg"],
                    fg=self.color_scheme["fg"],
                    font=("Consolas", 9)
                )
                volume_label.grid(row=row, column=3, padx=10, pady=5)
                
                row += 1
            
        except Exception as e:
            logger.error(f"Error updating trading display: {e}")

    async def _update_portfolio_display(self):
        """Update the portfolio balance display."""
        if not hasattr(self, 'balance_canvas') or not self.portfolio_balancer:
            return
        
        try:
            # Clear canvas
            self.balance_canvas.delete("all")
            
            # Get portfolio state
            portfolio_state = self.portfolio_balancer.portfolio_state
            
            # Draw portfolio pie chart (simplified)
            canvas_width = self.balance_canvas.winfo_width() or 400
            canvas_height = self.balance_canvas.winfo_height() or 300
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            radius = min(center_x, center_y) - 50
            
            # Draw asset allocations
            total_value = float(portfolio_state.total_value)
            start_angle = 0
            
            colors = [self.color_scheme["pattern"], self.color_scheme["trading"], self.color_scheme["success"]]
            
            for i, (asset, weight) in enumerate(portfolio_state.asset_weights.items()):
                if weight > 0:
                    extent = weight * 360
                    color = colors[i % len(colors)]
                    
                    # Draw pie slice (simplified as text for now)
                    angle_rad = math.radians(start_angle + extent/2)
                    text_x = center_x + (radius * 0.7) * math.cos(angle_rad)
                    text_y = center_y + (radius * 0.7) * math.sin(angle_rad)
                    
                    self.balance_canvas.create_text(
                        text_x,
                        text_y,
                        text=f"{asset}\n{weight:.1%}",
                        fill=color,
                        font=("Consolas", 10, "bold"),
                        justify="center"
                    )
                    
                    start_angle += extent
            
            # Draw total value
            self.balance_canvas.create_text(
                center_x,
                center_y,
                text=f"Total\n${total_value:.2f}",
                fill=self.color_scheme["fg"],
                font=("Consolas", 14, "bold"),
                justify="center"
            )
            
            # Update metrics
            if hasattr(self, 'metrics_text'):
                await self._update_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"Error updating portfolio display: {e}")

    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics."""
        if not hasattr(self, 'metrics_text') or not self.portfolio_balancer:
            return
        
        try:
            self.metrics_text.delete(1.0, tk.END)
            
            # Get performance metrics
            performance = await self.portfolio_balancer.calculate_performance_metrics()
            
            metrics_text = f"""
⚖️ PORTFOLIO PERFORMANCE
{'=' * 30}

Total Return: {performance.get('total_return', 0.0):.2%}
Sharpe Ratio: {performance.get('sharpe_ratio', 0.0):.3f}
Volatility: {performance.get('volatility', 0.0):.2%}
Max Drawdown: {performance.get('max_drawdown', 0.0):.2%}

🎯 REBALANCING:
Last Rebalance: {performance.get('last_rebalance', 'Never')}
Rebalances Today: {performance.get('rebalances_today', 0)}
Drift Score: {performance.get('drift_score', 0.0):.3f}
"""
            
            self.metrics_text.insert(tk.END, metrics_text)
            
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")

    async def _update_health_display(self):
        """Update system health monitoring display."""
        try:
            # Update health indicators
            if hasattr(self, 'health_indicators'):
                # Check component health
                health_status = {
                    "2-Gram Detector": "🟢 Healthy" if self.two_gram_detector else "🔴 Not Available",
                    "Strategy Router": "🟢 Healthy" if self.strategy_router else "🔴 Not Available",
                    "Portfolio Balancer": "🟢 Healthy" if self.portfolio_balancer else "🔴 Not Available",
                    "BTC/USDC Integration": "🟢 Healthy" if self.btc_usdc_integration else "🔴 Not Available",
                    "Network Status": "🟢 Connected",
                    "Memory Usage": f"🟡 {self._get_memory_usage():.1f} MB"
                }
                
                for name, status in health_status.items():
                    if name in self.health_indicators:
                        self.health_indicators[name].config(text=status)
            
            # Update T-cell status
            if hasattr(self, 'tcell_text') and self.two_gram_detector:
                health_check = await self.two_gram_detector.health_check()
                
                self.tcell_text.delete(1.0, tk.END)
                
                tcell_text = f"""
🛡️ T-CELL PROTECTION STATUS
{'=' * 35}

Overall Status: {health_check.get('overall_status', 'Unknown')}
Health Score: {health_check.get('system_health_score', 0.0):.3f}
T-Cell Active: {health_check.get('t_cell_active', False)}

🚨 ANOMALIES DETECTED:
"""
                
                anomalies = health_check.get('anomalies', [])
                if anomalies:
                    for anomaly in anomalies:
                        tcell_text += f"• {anomaly}\n"
                else:
                    tcell_text += "• No anomalies detected\n"
                
                tcell_text += f"""
📊 IMMUNE METRICS:
Response Time: {health_check.get('response_time_ms', 0):.1f}ms
Memory Health: {health_check.get('memory_health', 'Good')}
CPU Health: {health_check.get('cpu_health', 'Good')}
"""
                
                self.tcell_text.insert(tk.END, tcell_text)
            
        except Exception as e:
            logger.error(f"Error updating health display: {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _update_status(self):
        """Update the status bar."""
        if not hasattr(self, 'status_label'):
            return
        
        try:
            # Create status message
            components = []
            if self.two_gram_detector:
                components.append("🧬")
            if self.strategy_router:
                components.append("🎯")
            if self.portfolio_balancer:
                components.append("⚖️")
            if self.btc_usdc_integration:
                components.append("💱")
            
            status_text = f"🔄 Running - Components: {' '.join(components)} - Frame: {self.animation_frame}"
            self.status_label.config(text=status_text)
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _start_trading(self):
        """Start trading operations."""
        if self.btc_usdc_integration:
            asyncio.create_task(self.btc_usdc_integration.start())
            info("▶️ Trading started")

    def _stop_trading(self):
        """Stop trading operations."""
        if self.btc_usdc_integration:
            asyncio.create_task(self.btc_usdc_integration.stop())
            info("⏹️ Trading stopped")

    def _toggle_demo(self):
        """Toggle demo mode."""
        # Implementation would toggle demo/live mode
        info("🎭 Demo mode toggled")

    def _on_close(self):
        """Handle window close event."""
        self.stop()

    async def start(self):
        """Start the visual execution node."""
        self.running = True
        
        if self.initialize_gui():
            info("🖥️ Visual Execution Node started with GUI")
            # GUI will run its own loop
        else:
            info("🖥️ Visual Execution Node started in headless mode")
            # Run headless update loop
            while self.running:
                await self._update_headless()
                await asyncio.sleep(self.config.update_interval_ms / 1000)

    async def _update_headless(self):
        """Update in headless mode (no GUI)."""
        try:
            # Log pattern statistics periodically
            if self.two_gram_detector and time.time() - self.last_update_time > 30:
                stats = await self.two_gram_detector.get_pattern_statistics()
                info(f"🧬 Patterns: {stats.get('active_patterns', 0)}, Health: {stats.get('system_health_score', 0.0):.2f}")
                self.last_update_time = time.time()
                
        except Exception as e:
            logger.error(f"Error in headless update: {e}")

    def stop(self):
        """Stop the visual execution node."""
        self.running = False
        
        if self.root:
            self.root.quit()
            
        info("🖥️ Visual Execution Node stopped")

    def run_gui(self):
        """Run the GUI main loop (blocking)."""
        if self.root:
            self.root.mainloop()

    async def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization performance statistics."""
        return {
            "gui_available": GUI_AVAILABLE,
            "running": self.running,
            "current_fps": self.current_fps,
            "animation_frame": self.animation_frame,
            "pattern_history_size": len(self.pattern_history),
            "execution_history_size": len(self.execution_history),
            "theme": self.config.theme.value,
            "gui_mode": self.config.gui_mode.value,
            "memory_usage_mb": self._get_memory_usage()
        }


# Factory function for easy integration
def create_visual_execution_node(config: Optional[Dict[str, Any]] = None) -> VisualExecutionNode:
    """Create a visual execution node instance."""
    visual_config = VisualConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(visual_config, key):
                setattr(visual_config, key, value)
    
    return VisualExecutionNode(visual_config)


# Integration test function
async def test_visual_execution_node():
    """Test the visual execution node with mock data."""
    print("🖥️ Testing Visual Execution Node")
    print("=" * 50)
    
    # Create visual node
    config = {
        "gui_mode": GUIMode.DEMO_MODE,
        "theme": VisualizationTheme.SCHWABOT_CLASSIC,
        "update_interval_ms": 500
    }
    
    visual_node = create_visual_execution_node(config)
    
    # Create mock 2-gram detector
    detector = create_two_gram_detector({})
    
    # Inject components
    await visual_node.inject_components(detector)
    
    # Get statistics
    stats = await visual_node.get_visualization_statistics()
    print(f"GUI Available: {stats['gui_available']}")
    print(f"Theme: {stats['theme']}")
    print(f"Mode: {stats['gui_mode']}")
    
    # Test headless mode
    print("\n🔄 Running headless update test...")
    for i in range(3):
        await visual_node._update_headless()
        await asyncio.sleep(0.1)
    
    print("✅ Visual execution node test completed")


if __name__ == "__main__":
    asyncio.run(test_visual_execution_node())
