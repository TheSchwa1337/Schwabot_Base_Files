#!/usr/bin/env python3
"""
Schwabot Main Dashboard
======================

The primary user interface for Schwabot that demonstrates clean integration
of all mathematical systems through the UI State Bridge.

This is the "final destination" UI that all mathematical changes adapt to,
rather than constantly changing the UI to fit new math.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import dearpygui.dearpygui as dpg
import numpy as np

# Add core to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import state bridge and core systems
from ui_state_bridge import (
    UIStateBridge, create_ui_bridge, SystemStatus, AlertLevel,
    UISystemHealth, UISustainmentRadar, UIHardwareState, UITradingState
)

# Import core controllers (with fallbacks for demo)
try:
    from sustainment_underlay_controller import SustainmentUnderlayController
    from thermal_zone_manager import ThermalZoneManager
    from profit_navigator import AntiPoleProfitNavigator
    from fractal_core import FractalCore
    DEMO_MODE = False
except ImportError:
    DEMO_MODE = True
    print("⚠️ Running in demo mode - some controllers not available")

logger = logging.getLogger(__name__)

class SchwabotDashboard:
    """
    Main Schwabot Dashboard
    
    Provides a clean, intuitive interface that reflects all mathematical
    systems and provides easy configuration and monitoring.
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        
        # UI state and bridge
        self.ui_bridge: Optional[UIStateBridge] = None
        self.current_ui_state: Dict[str, Any] = {}
        
        # Dashboard state
        self.dashboard_active = False
        self.update_thread = None
        
        # DearPyGui tags for easy updates
        self.ui_tags = {
            'system_status': 'tag_system_status',
            'sustainment_index': 'tag_sustainment_index',
            'profit_24h': 'tag_profit_24h',
            'trades_today': 'tag_trades_today',
            'thermal_status': 'tag_thermal_status',
            
            # Sustainment radar bars
            'radar_anticipation': 'tag_radar_anticipation',
            'radar_integration': 'tag_radar_integration',
            'radar_responsiveness': 'tag_radar_responsiveness',
            'radar_simplicity': 'tag_radar_simplicity',
            'radar_economy': 'tag_radar_economy',
            'radar_survivability': 'tag_radar_survivability',
            'radar_continuity': 'tag_radar_continuity',
            'radar_improvisation': 'tag_radar_improvisation',
            
            # Hardware monitoring
            'cpu_usage': 'tag_cpu_usage',
            'gpu_usage': 'tag_gpu_usage',
            'cpu_temp': 'tag_cpu_temp',
            'gpu_temp': 'tag_gpu_temp',
            
            # Trading state
            'total_equity': 'tag_total_equity',
            'unrealized_pnl': 'tag_unrealized_pnl',
            'win_rate': 'tag_win_rate',
            'active_strategies': 'tag_active_strategies',
            
            # Alerts and logs
            'alert_list': 'tag_alert_list',
            'log_output': 'tag_log_output'
        }
        
        # Colors for different states
        self.status_colors = {
            SystemStatus.OPTIMAL: [0, 255, 0],      # Green
            SystemStatus.OPERATIONAL: [255, 255, 0], # Yellow
            SystemStatus.DEGRADED: [255, 165, 0],    # Orange
            SystemStatus.CRITICAL: [255, 0, 0],      # Red
            SystemStatus.OFFLINE: [128, 128, 128]    # Gray
        }
        
        logger.info("Schwabot Dashboard initialized")

    def create_ui_bridge_demo(self) -> UIStateBridge:
        """Create UI bridge with mock controllers for demo"""
        from demo_sustainment_underlay_integration import create_mock_controllers
        
        thermal_manager, cooldown_manager, profit_navigator, fractal_core = create_mock_controllers()
        
        # Create sustainment underlay
        sustainment_controller = SustainmentUnderlayController(
            thermal_manager=thermal_manager,
            cooldown_manager=cooldown_manager,
            profit_navigator=profit_navigator,
            fractal_core=fractal_core
        )
        
        # Start sustainment monitoring
        sustainment_controller.start_continuous_synthesis(interval=3.0)
        
        # Create UI bridge
        return create_ui_bridge(
            sustainment_controller, thermal_manager, profit_navigator, fractal_core
        )

    def initialize_dashboard(self) -> None:
        """Initialize the complete dashboard"""
        
        # Create UI bridge (demo mode if needed)
        if DEMO_MODE:
            self.ui_bridge = self.create_ui_bridge_demo()
        else:
            # Would create with real controllers
            pass
        
        # Register for UI state updates
        if self.ui_bridge:
            self.ui_bridge.register_ui_callback(self._on_ui_state_update)
        
        # Create DearPyGui context and main window
        dpg.create_context()
        self._create_main_window()
        
        # Setup viewport
        dpg.create_viewport(
            title="Schwabot v1.0 - Law of Sustainment Trading Platform",
            width=1400,
            height=900,
            resizable=True
        )
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        # Start update thread
        self.dashboard_active = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Schwabot Dashboard fully initialized and running")

    def _create_main_window(self) -> None:
        """Create the main dashboard window with all panels"""
        
        with dpg.window(label="Schwabot Dashboard", tag="main_window", 
                       width=1380, height=880, no_close=True, no_collapse=True):
            
            # Header section - system overview
            self._create_header_section()
            
            dpg.add_separator()
            
            # Main content - tabbed interface
            with dpg.tab_bar(label="Main Tabs"):
                
                # Overview tab - key metrics and status
                with dpg.tab(label="🏠 Overview"):
                    self._create_overview_tab()
                
                # Sustainment tab - 8-principle radar and details
                with dpg.tab(label="⚖️ Sustainment"):
                    self._create_sustainment_tab()
                
                # Trading tab - positions, performance, strategies
                with dpg.tab(label="💰 Trading"):
                    self._create_trading_tab()
                
                # Hardware tab - thermal, GPU, system monitoring
                with dpg.tab(label="🖥️ Hardware"):
                    self._create_hardware_tab()
                
                # Visualizer tab - Tesseract and advanced views
                with dpg.tab(label="📊 Visualizer"):
                    self._create_visualizer_tab()
                
                # Settings tab - configuration and setup
                with dpg.tab(label="⚙️ Settings"):
                    self._create_settings_tab()
                
                # Logs tab - alerts, logs, debugging
                with dpg.tab(label="📋 Logs"):
                    self._create_logs_tab()

    def _create_header_section(self) -> None:
        """Create header with key system status indicators"""
        
        with dpg.group(horizontal=True):
            # System status indicator
            with dpg.group():
                dpg.add_text("System Status:")
                dpg.add_text("INITIALIZING", tag=self.ui_tags['system_status'])
            
            dpg.add_spacer(width=50)
            
            # Key metrics
            with dpg.group():
                dpg.add_text("Sustainment Index:")
                dpg.add_text("0.500", tag=self.ui_tags['sustainment_index'])
            
            dpg.add_spacer(width=50)
            
            with dpg.group():
                dpg.add_text("24h Profit:")
                dpg.add_text("$0.00", tag=self.ui_tags['profit_24h'])
            
            dpg.add_spacer(width=50)
            
            with dpg.group():
                dpg.add_text("Trades Today:")
                dpg.add_text("0", tag=self.ui_tags['trades_today'])
            
            dpg.add_spacer(width=50)
            
            with dpg.group():
                dpg.add_text("Thermal Status:")
                dpg.add_text("unknown", tag=self.ui_tags['thermal_status'])

    def _create_overview_tab(self) -> None:
        """Create overview tab with key dashboard elements"""
        
        with dpg.group(horizontal=True):
            
            # Left column - Sustainment radar (compact)
            with dpg.group():
                dpg.add_text("Sustainment Radar (Live)", color=[100, 200, 255])
                self._create_compact_sustainment_radar()
            
            dpg.add_spacer(width=50)
            
            # Right column - Key metrics and alerts
            with dpg.group():
                dpg.add_text("Key Metrics", color=[100, 200, 255])
                
                with dpg.table(header_row=False, borders_innerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    
                    with dpg.table_row():
                        dpg.add_text("Total Equity:")
                        dpg.add_text("$0.00", tag=self.ui_tags['total_equity'])
                    
                    with dpg.table_row():
                        dpg.add_text("Unrealized P&L:")
                        dpg.add_text("$0.00", tag=self.ui_tags['unrealized_pnl'])
                    
                    with dpg.table_row():
                        dpg.add_text("Win Rate:")
                        dpg.add_text("0%", tag=self.ui_tags['win_rate'])
                    
                    with dpg.table_row():
                        dpg.add_text("CPU Usage:")
                        dpg.add_text("0%", tag=self.ui_tags['cpu_usage'])
                    
                    with dpg.table_row():
                        dpg.add_text("GPU Usage:")
                        dpg.add_text("0%", tag=self.ui_tags['gpu_usage'])
        
        dpg.add_separator()
        
        # Recent alerts section
        dpg.add_text("Recent Alerts", color=[255, 200, 100])
        dpg.add_listbox([], label="", tag=self.ui_tags['alert_list'], num_items=5)

    def _create_compact_sustainment_radar(self) -> None:
        """Create compact sustainment radar with progress bars"""
        
        principles = [
            ('Anticipation', 'radar_anticipation'),
            ('Integration', 'radar_integration'),
            ('Responsiveness', 'radar_responsiveness'),
            ('Simplicity', 'radar_simplicity'),
            ('Economy', 'radar_economy'),
            ('Survivability', 'radar_survivability'),
            ('Continuity', 'radar_continuity'),
            ('Improvisation', 'radar_improvisation')
        ]
        
        for principle, tag_name in principles:
            dpg.add_text(f"{principle}:")
            dpg.add_progress_bar(
                default_value=0.5,
                tag=self.ui_tags[tag_name],
                width=200
            )

    def _create_sustainment_tab(self) -> None:
        """Create detailed sustainment monitoring tab"""
        
        dpg.add_text("Law of Sustainment - 8 Principle Framework", color=[100, 255, 100])
        dpg.add_text("Mathematical synthesis ensuring sustainable operation")
        
        dpg.add_separator()
        
        # Detailed principle breakdown
        with dpg.table(header_row=True, borders_innerV=True, borders_innerH=True):
            dpg.add_table_column(label="Principle")
            dpg.add_table_column(label="Value")
            dpg.add_table_column(label="Status")
            dpg.add_table_column(label="Description")
            
            principles = [
                ("Anticipation", "radar_anticipation", "Predictive modeling capability"),
                ("Integration", "radar_integration", "System coherence and harmony"),
                ("Responsiveness", "radar_responsiveness", "Real-time adaptation speed"),
                ("Simplicity", "radar_simplicity", "Complexity management"),
                ("Economy", "radar_economy", "Resource efficiency & profit optimization"),
                ("Survivability", "radar_survivability", "Risk management & resilience"),
                ("Continuity", "radar_continuity", "Persistent operation capability"),
                ("Improvisation", "radar_improvisation", "Creative adaptation ability")
            ]
            
            for principle, tag_name, description in principles:
                with dpg.table_row():
                    dpg.add_text(principle)
                    dpg.add_text("0.500", tag=f"detail_{tag_name}")
                    dpg.add_text("NORMAL", tag=f"status_{tag_name}")
                    dpg.add_text(description)

    def _create_trading_tab(self) -> None:
        """Create trading performance and strategy tab"""
        
        dpg.add_text("Trading Performance & Strategy Management", color=[100, 255, 100])
        
        with dpg.group(horizontal=True):
            
            # Portfolio overview
            with dpg.group():
                dpg.add_text("Portfolio Overview", color=[200, 200, 255])
                
                dpg.add_text("Active Strategies:")
                dpg.add_text("Loading...", tag=self.ui_tags['active_strategies'])
                
                dpg.add_separator()
                
                dpg.add_text("Strategy Performance:")
                with dpg.table(header_row=True):
                    dpg.add_table_column(label="Strategy")
                    dpg.add_table_column(label="Return")
                    dpg.add_table_column(label="Trades")
                    dpg.add_table_column(label="Win Rate")
                    
                    # Strategy rows will be populated dynamically
            
            dpg.add_spacer(width=50)
            
            # Performance charts placeholder
            with dpg.group():
                dpg.add_text("Performance Charts", color=[200, 200, 255])
                dpg.add_text("📊 Profit/Loss Chart")
                dpg.add_text("📈 Equity Curve")
                dpg.add_text("📉 Drawdown Analysis")
                dpg.add_text("⚡ Real-time P&L")

    def _create_hardware_tab(self) -> None:
        """Create hardware monitoring tab"""
        
        dpg.add_text("Hardware & Thermal Management", color=[100, 255, 100])
        
        with dpg.group(horizontal=True):
            
            # CPU monitoring
            with dpg.group():
                dpg.add_text("CPU Monitoring", color=[255, 200, 200])
                
                dpg.add_text("CPU Usage:")
                dpg.add_progress_bar(default_value=0.3, tag=f"progress_{self.ui_tags['cpu_usage']}")
                
                dpg.add_text("CPU Temperature:")
                dpg.add_text("55°C", tag=self.ui_tags['cpu_temp'])
            
            dpg.add_spacer(width=50)
            
            # GPU monitoring
            with dpg.group():
                dpg.add_text("GPU Monitoring", color=[200, 255, 200])
                
                dpg.add_text("GPU Usage:")
                dpg.add_progress_bar(default_value=0.4, tag=f"progress_{self.ui_tags['gpu_usage']}")
                
                dpg.add_text("GPU Temperature:")
                dpg.add_text("65°C", tag=self.ui_tags['gpu_temp'])
        
        dpg.add_separator()
        
        # Thermal zone management
        dpg.add_text("Thermal Zone Management", color=[255, 255, 200])
        dpg.add_text("Current thermal state and cooling strategies")

    def _create_visualizer_tab(self) -> None:
        """Create visualizer integration tab"""
        
        dpg.add_text("Tesseract Visualizer & Advanced Analytics", color=[100, 255, 100])
        
        dpg.add_button(label="🔳 Open Tesseract Visualizer", 
                      callback=self._open_tesseract_visualizer)
        
        dpg.add_button(label="📊 Open Advanced Tesseract", 
                      callback=self._open_advanced_tesseract)
        
        dpg.add_separator()
        
        dpg.add_text("Current Visualization State:")
        dpg.add_text("• Fractal Coherence: 0.600")
        dpg.add_text("• Pattern Strength: 0.750")
        dpg.add_text("• Active Overlays: 3")
        dpg.add_text("• Glyph Count: 12")

    def _create_settings_tab(self) -> None:
        """Create settings and configuration tab"""
        
        dpg.add_text("Schwabot Configuration", color=[100, 255, 100])
        
        # API Configuration section
        with dpg.collapsing_header(label="API Configuration"):
            
            dpg.add_input_text(label="Exchange API Key", password=True)
            dpg.add_input_text(label="Exchange Secret", password=True)
            dpg.add_combo(label="Exchange", items=["Coinbase Pro", "Binance", "Kraken"])
            
            dpg.add_button(label="Test Connection", callback=self._test_api_connection)
        
        # Strategy Configuration
        with dpg.collapsing_header(label="Strategy Configuration"):
            
            dpg.add_checkbox(label="Enable Momentum Strategy")
            dpg.add_checkbox(label="Enable Reversal Strategy")
            dpg.add_checkbox(label="Enable Anti-Pole Strategy")
            
            dpg.add_slider_float(label="Risk Level", default_value=0.5, 
                                min_value=0.1, max_value=1.0)
        
        # Sustainment Thresholds
        with dpg.collapsing_header(label="Sustainment Thresholds"):
            
            dpg.add_slider_float(label="Critical Sustainment Index", 
                                default_value=0.65, min_value=0.1, max_value=1.0)
            dpg.add_slider_float(label="Survivability Threshold", 
                                default_value=0.85, min_value=0.1, max_value=1.0)

    def _create_logs_tab(self) -> None:
        """Create logs and debugging tab"""
        
        dpg.add_text("System Logs & Debugging", color=[100, 255, 100])
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Clear Logs", callback=self._clear_logs)
            dpg.add_button(label="Export Logs", callback=self._export_logs)
            dpg.add_button(label="Refresh", callback=self._refresh_logs)
        
        dpg.add_separator()
        
        # Log output area
        dpg.add_input_text(
            label="",
            tag=self.ui_tags['log_output'],
            multiline=True,
            readonly=True,
            height=300,
            width=-1
        )

    def _on_ui_state_update(self, ui_state: Dict[str, Any]) -> None:
        """Handle UI state updates from the bridge"""
        self.current_ui_state = ui_state

    def _update_loop(self) -> None:
        """Main update loop for dashboard"""
        while self.dashboard_active:
            try:
                if self.current_ui_state:
                    self._update_dashboard_values()
                time.sleep(1.0)  # Update every second
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(1.0)

    def _update_dashboard_values(self) -> None:
        """Update all dashboard values from current UI state"""
        
        state = self.current_ui_state
        
        # Update header status
        system_health = state.get('system_health', {})
        dpg.set_value(self.ui_tags['system_status'], system_health.get('status', 'unknown').upper())
        dpg.set_value(self.ui_tags['sustainment_index'], f"{system_health.get('sustainment_index', 0.5):.3f}")
        dpg.set_value(self.ui_tags['profit_24h'], f"${system_health.get('profit_24h', 0.0):.2f}")
        dpg.set_value(self.ui_tags['trades_today'], str(system_health.get('trades_today', 0)))
        dpg.set_value(self.ui_tags['thermal_status'], system_health.get('thermal_status', 'unknown'))
        
        # Update sustainment radar
        radar = state.get('sustainment_radar', {})
        for principle in ['anticipation', 'integration', 'responsiveness', 'simplicity',
                         'economy', 'survivability', 'continuity', 'improvisation']:
            value = radar.get(principle, 0.5)
            dpg.set_value(self.ui_tags[f'radar_{principle}'], value)
        
        # Update hardware state
        hardware = state.get('hardware_state', {})
        dpg.set_value(self.ui_tags['cpu_usage'], f"{hardware.get('cpu_usage', 0.0) * 100:.1f}%")
        dpg.set_value(self.ui_tags['gpu_usage'], f"{hardware.get('gpu_usage', 0.0) * 100:.1f}%")
        dpg.set_value(self.ui_tags['cpu_temp'], f"{hardware.get('cpu_temp', 0.0):.0f}°C")
        dpg.set_value(self.ui_tags['gpu_temp'], f"{hardware.get('gpu_temp', 0.0):.0f}°C")
        
        # Update trading state
        trading = state.get('trading_state', {})
        dpg.set_value(self.ui_tags['total_equity'], f"${trading.get('total_equity', 0.0):.2f}")
        dpg.set_value(self.ui_tags['unrealized_pnl'], f"${trading.get('unrealized_pnl', 0.0):.2f}")
        dpg.set_value(self.ui_tags['win_rate'], f"{trading.get('win_rate', 0.0) * 100:.1f}%")
        
        active_strategies = trading.get('active_strategies', [])
        dpg.set_value(self.ui_tags['active_strategies'], ", ".join(active_strategies))
        
        # Update alerts
        alerts = state.get('alerts', [])
        alert_items = [f"{alert['level']}: {alert['title']}" for alert in alerts[-5:]]
        dpg.configure_item(self.ui_tags['alert_list'], items=alert_items)

    # Button callbacks
    def _open_tesseract_visualizer(self):
        logger.info("Opening Tesseract Visualizer...")
        # Would launch the tesseract visualizer
    
    def _open_advanced_tesseract(self):
        logger.info("Opening Advanced Tesseract Visualizer...")
        # Would launch the advanced tesseract visualizer
    
    def _test_api_connection(self):
        logger.info("Testing API connection...")
        # Would test the API connection
    
    def _clear_logs(self):
        dpg.set_value(self.ui_tags['log_output'], "")
    
    def _export_logs(self):
        logger.info("Exporting logs...")
        # Would export logs to file
    
    def _refresh_logs(self):
        # Would refresh log display
        pass

    def run(self) -> None:
        """Run the dashboard"""
        
        try:
            self.initialize_dashboard()
            
            # Main render loop
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
                
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the dashboard cleanly"""
        
        self.dashboard_active = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        if self.ui_bridge:
            self.ui_bridge.stop_ui_monitoring()
        
        dpg.destroy_context()
        
        logger.info("Schwabot Dashboard shutdown complete")

def main():
    """Main entry point for Schwabot Dashboard"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Schwabot v1.0 Dashboard")
    print("=" * 60)
    print("Law of Sustainment Trading Platform")
    print("Mathematical synthesis for sustainable profit")
    print("=" * 60)
    
    try:
        dashboard = SchwabotDashboard()
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 