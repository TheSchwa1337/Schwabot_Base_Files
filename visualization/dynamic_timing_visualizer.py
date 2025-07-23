#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ DYNAMIC TIMING VISUALIZER - HUMAN-FRIENDLY INTERFACE
======================================================

Advanced visualization system for the dynamic timing components:
- Rolling profit calculations with real-time charts
- Market regime detection with visual indicators
- Timing triggers with event logs
- Performance metrics with live updates
- Human-friendly explanations and recommendations

Features:
- Real-time charts and graphs
- Interactive controls
- Performance dashboards
- System status monitoring
- Trading recommendations
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import our dynamic timing systems
try:
    from core.dynamic_timing_system import get_dynamic_timing_system, TimingRegime, OrderTiming
    from core.enhanced_real_time_data_puller import get_enhanced_data_puller, DataSource
    DYNAMIC_TIMING_AVAILABLE = True
except ImportError:
    DYNAMIC_TIMING_AVAILABLE = False
    print("Warning: Dynamic timing system not available")

class DynamicTimingVisualizer:
    """Comprehensive visualizer for dynamic timing system."""
    
    def __init__(self):
        """Initialize the dynamic timing visualizer."""
        self.root = tk.Tk()
        self.root.title("⚡ Schwabot Dynamic Timing Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize systems
        if DYNAMIC_TIMING_AVAILABLE:
            self.dynamic_timing = get_dynamic_timing_system()
            self.data_puller = get_enhanced_data_puller()
        else:
            self.dynamic_timing = None
            self.data_puller = None
        
        # Data storage for visualization
        self.profit_data = []
        self.volatility_data = []
        self.momentum_data = []
        self.regime_data = []
        self.timing_events = []
        self.performance_metrics = {}
        
        # Animation and update control
        self.animation_running = False
        self.update_thread = None
        
        # Setup GUI
        self.setup_gui()
        self.setup_callbacks()
        
        # Start data collection
        self.start_data_collection()
    
    def setup_gui(self):
        """Setup the main GUI interface."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            title_frame, 
            text="⚡ SCHWABOT DYNAMIC TIMING SYSTEM", 
            font=('Arial', 18, 'bold')
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Rolling Measurements • Timing Triggers • Regime Detection • Performance Optimization",
            font=('Arial', 10)
        )
        subtitle_label.pack()
        
        # Control panel
        self.setup_control_panel(main_container)
        
        # Main visualization area
        self.setup_visualization_area(main_container)
        
        # Status bar
        self.setup_status_bar(main_container)
    
    def setup_control_panel(self, parent):
        """Setup control panel with system controls."""
        control_frame = ttk.LabelFrame(parent, text="System Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # System status
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="System Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Initializing...", foreground="orange")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="🚀 Start System", 
            command=self.start_system
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="🛑 Stop System", 
            command=self.stop_system,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_button = ttk.Button(
            button_frame, 
            text="🔄 Reset Data", 
            command=self.reset_data
        )
        self.reset_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Settings
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Update Interval (ms):").pack(side=tk.LEFT)
        self.update_interval_var = tk.StringVar(value="1000")
        update_entry = ttk.Entry(settings_frame, textvariable=self.update_interval_var, width=10)
        update_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(settings_frame, text="Data Points:").pack(side=tk.LEFT)
        self.data_points_var = tk.StringVar(value="100")
        points_entry = ttk.Entry(settings_frame, textvariable=self.data_points_var, width=10)
        points_entry.pack(side=tk.LEFT, padx=(5, 0))
    
    def setup_visualization_area(self, parent):
        """Setup the main visualization area with charts and metrics."""
        # Create notebook for different views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Real-time Performance Dashboard
        self.setup_performance_dashboard()
        
        # Tab 2: Regime Detection & Timing
        self.setup_regime_detection_tab()
        
        # Tab 3: Rolling Metrics Analysis
        self.setup_rolling_metrics_tab()
        
        # Tab 4: System Status & Events
        self.setup_system_status_tab()
        
        # Tab 5: Trading Recommendations
        self.setup_recommendations_tab()
    
    def setup_performance_dashboard(self):
        """Setup real-time performance dashboard."""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Performance Dashboard")
        
        # Create matplotlib figure for real-time charts
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor('#1e1e1e')
        
        # Profit chart
        self.axes[0, 0].set_title('Rolling Profit', color='white', fontsize=12)
        self.axes[0, 0].set_facecolor('#2d2d2d')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.profit_line, = self.axes[0, 0].plot([], [], 'g-', linewidth=2, label='Profit')
        self.axes[0, 0].legend()
        
        # Volatility chart
        self.axes[0, 1].set_title('Market Volatility', color='white', fontsize=12)
        self.axes[0, 1].set_facecolor('#2d2d2d')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.volatility_line, = self.axes[0, 1].plot([], [], 'r-', linewidth=2, label='Volatility')
        self.axes[0, 1].legend()
        
        # Momentum chart
        self.axes[1, 0].set_title('Market Momentum', color='white', fontsize=12)
        self.axes[1, 0].set_facecolor('#2d2d2d')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.momentum_line, = self.axes[1, 0].plot([], [], 'b-', linewidth=2, label='Momentum')
        self.axes[1, 0].legend()
        
        # Regime indicator
        self.axes[1, 1].set_title('Market Regime', color='white', fontsize=12)
        self.axes[1, 1].set_facecolor('#2d2d2d')
        self.regime_text = self.axes[1, 1].text(0.5, 0.5, 'NORMAL', 
                                               ha='center', va='center', 
                                               fontsize=20, fontweight='bold',
                                               color='yellow')
        self.axes[1, 1].set_xlim(0, 1)
        self.axes[1, 1].set_ylim(0, 1)
        self.axes[1, 1].axis('off')
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, dashboard_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Performance metrics panel
        metrics_frame = ttk.LabelFrame(dashboard_frame, text="Live Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create metrics display
        self.metrics_labels = {}
        metrics = [
            ('rolling_profit', 'Rolling Profit:'),
            ('timing_accuracy', 'Timing Accuracy:'),
            ('current_regime', 'Current Regime:'),
            ('total_signals', 'Total Signals:'),
            ('successful_signals', 'Successful Signals:'),
            ('avg_execution_time', 'Avg Execution Time:'),
            ('data_pull_rate', 'Data Pull Rate:'),
            ('system_uptime', 'System Uptime:')
        ]
        
        for i, (key, label) in enumerate(metrics):
            row = i // 4
            col = i % 4
            
            ttk.Label(metrics_frame, text=label).grid(row=row, column=col*2, sticky=tk.W, padx=5)
            self.metrics_labels[key] = ttk.Label(metrics_frame, text="0.00", font=('Arial', 10, 'bold'))
            self.metrics_labels[key].grid(row=row, column=col*2+1, sticky=tk.W, padx=5)
    
    def setup_regime_detection_tab(self):
        """Setup regime detection and timing analysis tab."""
        regime_frame = ttk.Frame(self.notebook)
        self.notebook.add(regime_frame, text="🔄 Regime Detection")
        
        # Regime history chart
        regime_chart_frame = ttk.Frame(regime_frame)
        regime_chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.regime_fig, self.regime_ax = plt.subplots(figsize=(10, 6))
        self.regime_fig.patch.set_facecolor('#1e1e1e')
        self.regime_ax.set_facecolor('#2d2d2d')
        self.regime_ax.set_title('Regime History', color='white', fontsize=14)
        self.regime_ax.grid(True, alpha=0.3)
        
        self.regime_canvas = FigureCanvasTkAgg(self.regime_fig, regime_chart_frame)
        self.regime_canvas.draw()
        self.regime_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Regime information panel
        info_frame = ttk.LabelFrame(regime_frame, text="Regime Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Regime details
        self.regime_info_text = tk.Text(info_frame, height=8, width=80, bg='#2d2d2d', fg='white')
        self.regime_info_text.pack(fill=tk.X)
        
        # Timing triggers panel
        triggers_frame = ttk.LabelFrame(regime_frame, text="Timing Triggers", padding="10")
        triggers_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.triggers_text = tk.Text(triggers_frame, height=6, width=80, bg='#2d2d2d', fg='white')
        self.triggers_text.pack(fill=tk.X)
    
    def setup_rolling_metrics_tab(self):
        """Setup rolling metrics analysis tab."""
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="📈 Rolling Metrics")
        
        # Metrics analysis chart
        analysis_frame = ttk.Frame(metrics_frame)
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.metrics_fig, self.metrics_axes = plt.subplots(2, 2, figsize=(12, 8))
        self.metrics_fig.patch.set_facecolor('#1e1e1e')
        
        # Setup subplots
        titles = ['Profit Distribution', 'Volatility Trends', 'Momentum Analysis', 'Performance Correlation']
        for i, ax in enumerate(self.metrics_axes.flat):
            ax.set_title(titles[i], color='white', fontsize=12)
            ax.set_facecolor('#2d2d2d')
            ax.grid(True, alpha=0.3)
        
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, analysis_frame)
        self.metrics_canvas.draw()
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Metrics summary
        summary_frame = ttk.LabelFrame(metrics_frame, text="Metrics Summary", padding="10")
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.summary_text = tk.Text(summary_frame, height=6, width=80, bg='#2d2d2d', fg='white')
        self.summary_text.pack(fill=tk.X)
    
    def setup_system_status_tab(self):
        """Setup system status and events tab."""
        status_frame = ttk.Frame(self.notebook)
        self.notebook.add(status_frame, text="⚙️ System Status")
        
        # System health indicators
        health_frame = ttk.LabelFrame(status_frame, text="System Health", padding="10")
        health_frame.pack(fill=tk.X, pady=10)
        
        # Health indicators
        self.health_indicators = {}
        health_items = [
            ('dynamic_timing', 'Dynamic Timing System'),
            ('data_puller', 'Data Puller System'),
            ('regime_detection', 'Regime Detection'),
            ('timing_triggers', 'Timing Triggers'),
            ('rolling_metrics', 'Rolling Metrics'),
            ('performance_tracking', 'Performance Tracking')
        ]
        
        for i, (key, label) in enumerate(health_items):
            row = i // 3
            col = i % 3
            
            ttk.Label(health_frame, text=label).grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=2)
            self.health_indicators[key] = ttk.Label(health_frame, text="❌", foreground="red")
            self.health_indicators[key].grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=2)
        
        # Event log
        events_frame = ttk.LabelFrame(status_frame, text="System Events", padding="10")
        events_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Event log with scrollbar
        log_frame = ttk.Frame(events_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.event_log = tk.Text(log_frame, bg='#2d2d2d', fg='white', font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.event_log.yview)
        self.event_log.configure(yscrollcommand=scrollbar.set)
        
        self.event_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_recommendations_tab(self):
        """Setup trading recommendations tab."""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="💡 Trading Recommendations")
        
        # Current recommendation
        current_frame = ttk.LabelFrame(rec_frame, text="Current Recommendation", padding="10")
        current_frame.pack(fill=tk.X, pady=10)
        
        self.recommendation_text = tk.Text(current_frame, height=8, width=80, bg='#2d2d2d', fg='white')
        self.recommendation_text.pack(fill=tk.X)
        
        # Recommendation history
        history_frame = ttk.LabelFrame(rec_frame, text="Recommendation History", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.recommendation_history = tk.Text(history_frame, bg='#2d2d2d', fg='white', font=('Consolas', 9))
        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.recommendation_history.yview)
        self.recommendation_history.configure(yscrollcommand=history_scrollbar.set)
        
        self.recommendation_history.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_status_bar(self, parent):
        """Setup status bar."""
        self.status_var = tk.StringVar(value="Ready to start dynamic timing visualization...")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def setup_callbacks(self):
        """Setup system callbacks."""
        if DYNAMIC_TIMING_AVAILABLE and self.dynamic_timing:
            # Set up callbacks for dynamic timing system
            self.dynamic_timing.set_data_pull_callback(self._on_pull_interval_adjustment)
            self.dynamic_timing.set_order_execution_callback(self._on_order_timing_adjustment)
            self.dynamic_timing.set_regime_change_callback(self._on_regime_change)
    
    def _on_pull_interval_adjustment(self, new_interval):
        """Handle pull interval adjustment."""
        self.add_event(f"⚡ Pull interval adjusted to {new_interval:.3f}s")
    
    def _on_order_timing_adjustment(self, timing_strategy):
        """Handle order timing adjustment."""
        self.add_event(f"🚀 Order timing strategy: {timing_strategy.value}")
    
    def _on_regime_change(self, old_regime, new_regime, volatility, momentum):
        """Handle regime change."""
        self.add_event(f"🔄 Regime change: {old_regime.value} → {new_regime.value}")
        self.add_recommendation(f"Market regime changed from {old_regime.value} to {new_regime.value}. "
                              f"Volatility: {volatility:.4f}, Momentum: {momentum:.4f}")
    
    def start_system(self):
        """Start the dynamic timing system."""
        try:
            if DYNAMIC_TIMING_AVAILABLE and self.dynamic_timing:
                success = self.dynamic_timing.start()
                if success:
                    self.status_label.config(text="Running", foreground="green")
                    self.start_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.NORMAL)
                    self.add_event("🚀 Dynamic timing system started")
                else:
                    messagebox.showerror("Error", "Failed to start dynamic timing system")
            else:
                # Simulate system for demo
                self.status_label.config(text="Demo Mode", foreground="blue")
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.add_event("🎮 Started in demo mode")
            
            # Start animation
            self.start_animation()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {e}")
    
    def stop_system(self):
        """Stop the dynamic timing system."""
        try:
            if DYNAMIC_TIMING_AVAILABLE and self.dynamic_timing:
                self.dynamic_timing.stop()
            
            self.status_label.config(text="Stopped", foreground="red")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Stop animation
            self.stop_animation()
            
            self.add_event("🛑 System stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop system: {e}")
    
    def reset_data(self):
        """Reset all data."""
        self.profit_data = []
        self.volatility_data = []
        self.momentum_data = []
        self.regime_data = []
        self.timing_events = []
        
        self.add_event("🔄 Data reset")
    
    def start_data_collection(self):
        """Start data collection thread."""
        self.update_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.update_thread.start()
    
    def _data_collection_loop(self):
        """Data collection loop."""
        while True:
            try:
                if self.dynamic_timing and self.dynamic_timing.active:
                    # Get real data from dynamic timing system
                    status = self.dynamic_timing.get_system_status()
                    self.performance_metrics = status
                    
                    # Add sample data for visualization
                    current_time = time.time()
                    self.profit_data.append((current_time, status.get('rolling_profit', 0)))
                    self.volatility_data.append((current_time, status.get('current_volatility', 0)))
                    self.momentum_data.append((current_time, status.get('current_momentum', 0)))
                    
                    # Limit data points
                    max_points = int(self.data_points_var.get())
                    if len(self.profit_data) > max_points:
                        self.profit_data = self.profit_data[-max_points:]
                        self.volatility_data = self.volatility_data[-max_points:]
                        self.momentum_data = self.momentum_data[-max_points:]
                
                time.sleep(0.1)  # 100ms update rate
                
            except Exception as e:
                print(f"Error in data collection: {e}")
                time.sleep(1.0)
    
    def start_animation(self):
        """Start matplotlib animation."""
        self.animation_running = True
        self.ani = animation.FuncAnimation(
            self.fig, self._update_charts, interval=int(self.update_interval_var.get()),
            blit=False, cache_frame_data=False
        )
    
    def stop_animation(self):
        """Stop matplotlib animation."""
        self.animation_running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
    
    def _update_charts(self, frame):
        """Update all charts."""
        try:
            if not self.profit_data:
                return
            
            # Update profit chart
            times, profits = zip(*self.profit_data)
            self.profit_line.set_data(times, profits)
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
            
            # Update volatility chart
            times, volatilities = zip(*self.volatility_data)
            self.volatility_line.set_data(times, volatilities)
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()
            
            # Update momentum chart
            times, momentums = zip(*self.momentum_data)
            self.momentum_line.set_data(times, momentums)
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
            
            # Update regime indicator
            if self.dynamic_timing:
                regime = self.dynamic_timing.get_current_regime()
                regime_colors = {
                    TimingRegime.CALM: 'green',
                    TimingRegime.NORMAL: 'yellow',
                    TimingRegime.VOLATILE: 'orange',
                    TimingRegime.EXTREME: 'red',
                    TimingRegime.CRISIS: 'purple'
                }
                color = regime_colors.get(regime, 'yellow')
                self.regime_text.set_text(regime.value.upper())
                self.regime_text.set_color(color)
            
            # Update metrics labels
            self._update_metrics_labels()
            
        except Exception as e:
            print(f"Error updating charts: {e}")
    
    def _update_metrics_labels(self):
        """Update metrics labels with current values."""
        try:
            if self.performance_metrics:
                self.metrics_labels['rolling_profit'].config(
                    text=f"{self.performance_metrics.get('rolling_profit', 0):.4f}"
                )
                self.metrics_labels['timing_accuracy'].config(
                    text=f"{self.performance_metrics.get('timing_accuracy', 0):.2f}"
                )
                self.metrics_labels['current_regime'].config(
                    text=self.performance_metrics.get('current_regime', 'unknown')
                )
                self.metrics_labels['total_signals'].config(
                    text=str(self.performance_metrics.get('total_signals', 0))
                )
                self.metrics_labels['successful_signals'].config(
                    text=str(self.performance_metrics.get('successful_signals', 0))
                )
                self.metrics_labels['avg_execution_time'].config(
                    text=f"{self.performance_metrics.get('avg_execution_time', 0):.3f}s"
                )
                self.metrics_labels['data_pull_rate'].config(
                    text=f"{self.performance_metrics.get('data_pull_rate', 0):.1f}/s"
                )
                self.metrics_labels['system_uptime'].config(
                    text=f"{self.performance_metrics.get('uptime', 0):.1f}s"
                )
                
                # Update health indicators
                self._update_health_indicators()
                
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def _update_health_indicators(self):
        """Update system health indicators."""
        try:
            if self.dynamic_timing:
                self.health_indicators['dynamic_timing'].config(text="✅", foreground="green")
            if self.data_puller:
                self.health_indicators['data_puller'].config(text="✅", foreground="green")
            
            # Update other indicators based on system status
            if self.performance_metrics.get('rolling_profit', 0) > 0:
                self.health_indicators['rolling_metrics'].config(text="✅", foreground="green")
            else:
                self.health_indicators['rolling_metrics'].config(text="⚠️", foreground="orange")
                
        except Exception as e:
            print(f"Error updating health indicators: {e}")
    
    def add_event(self, message):
        """Add event to the event log."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            event_message = f"[{timestamp}] {message}\n"
            
            self.event_log.insert(tk.END, event_message)
            self.event_log.see(tk.END)
            
            # Limit log size
            lines = self.event_log.get("1.0", tk.END).split('\n')
            if len(lines) > 100:
                self.event_log.delete("1.0", "50.0")
                
        except Exception as e:
            print(f"Error adding event: {e}")
    
    def add_recommendation(self, message):
        """Add trading recommendation."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            rec_message = f"[{timestamp}] {message}\n"
            
            self.recommendation_history.insert(tk.END, rec_message)
            self.recommendation_history.see(tk.END)
            
            # Update current recommendation
            self.recommendation_text.delete("1.0", tk.END)
            self.recommendation_text.insert("1.0", message)
            
        except Exception as e:
            print(f"Error adding recommendation: {e}")
    
    def run(self):
        """Run the visualizer."""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error running visualizer: {e}")

def main():
    """Main function to run the dynamic timing visualizer."""
    try:
        print("⚡ Starting Schwabot Dynamic Timing Visualizer...")
        visualizer = DynamicTimingVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"❌ Error starting visualizer: {e}")

if __name__ == "__main__":
    main() 