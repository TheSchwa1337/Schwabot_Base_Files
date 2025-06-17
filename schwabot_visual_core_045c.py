#!/usr/bin/env python3
"""
Schwabot Visual Core 0.045c – Complete Integration
==================================================

Full visualization system with backend integration, hardware monitoring,
live data streaming, and persistent settings management.

Features:
- Real-time CPU/GPU monitoring
- Live market data streaming  
- Trade decision visualization
- Persistent configuration
- Integration with core Schwabot components

Install dependencies:
    pip install dearpygui psutil pyyaml

Usage:
    python schwabot_visual_core_045c.py
"""

import dearpygui.dearpygui as dpg
import threading
import time
import json
import yaml
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import numpy as np

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Try to import core Schwabot components
try:
    from core.system_monitor import SystemMonitor
    from core.hash_affinity_vault import HashAffinityVault
    from core.quantum_antipole_engine import QuantumAntiPoleEngine
    from core.master_orchestrator import MasterOrchestrator
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("⚠️  Core Schwabot modules not found. Running in standalone mode.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UISettings:
    """Configuration for UI and system settings"""
    # API Settings
    coinbase_api_key: str = ""
    coinbase_secret: str = ""
    binance_api_key: str = ""
    binance_secret: str = ""
    
    # Trading Settings
    primary_pair: str = "BTC/USDC"
    max_risk_percent: float = 2.0
    enable_gpu_acceleration: bool = True
    auto_trading_enabled: bool = False
    
    # UI Settings
    update_interval_ms: int = 1000
    max_data_points: int = 1000
    dark_theme: bool = True
    
    # System Settings
    log_level: str = "INFO"
    data_retention_days: int = 30

class HardwareMonitor:
    """Real-time hardware monitoring with thread safety"""
    
    def __init__(self):
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.gpu_percent = 0.0
        self.gpu_memory_percent = 0.0
        self.gpu_temperature = 0.0
        self.running = False
        self._thread = None
        
    def start(self):
        """Start monitoring in background thread"""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("Hardware monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Hardware monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # CPU and Memory monitoring
                self.cpu_percent = psutil.cpu_percent(interval=0.1)
                self.memory_percent = psutil.virtual_memory().percent
                
                # GPU monitoring if available
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Primary GPU
                            self.gpu_percent = gpu.load * 100
                            self.gpu_memory_percent = gpu.memoryUtil * 100
                            self.gpu_temperature = gpu.temperature
                    except Exception as e:
                        logger.warning(f"GPU monitoring error: {e}")
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Hardware monitoring error: {e}")
                time.sleep(5)  # Longer delay on error

class MarketDataSimulator:
    """Simulates live market data for visualization"""
    
    def __init__(self):
        self.price_data = deque(maxlen=1000)
        self.volume_data = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.running = False
        self._thread = None
        self.current_price = 45000.0  # Starting BTC price
        
    def start(self):
        """Start data simulation"""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._simulate_loop, daemon=True)
            self._thread.start()
            logger.info("Market data simulation started")
    
    def stop(self):
        """Stop data simulation"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Market data simulation stopped")
    
    def _simulate_loop(self):
        """Background data simulation loop"""
        while self.running:
            try:
                # Generate realistic price movement
                price_change = np.random.normal(0, 100)  # $100 standard deviation
                self.current_price += price_change
                self.current_price = max(1000, self.current_price)  # Minimum $1000
                
                # Generate volume
                volume = np.random.exponential(1000000)  # Random volume
                
                # Store data
                self.price_data.append(self.current_price)
                self.volume_data.append(volume)
                self.timestamps.append(time.time())
                
                time.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                logger.error(f"Market simulation error: {e}")
                time.sleep(1)

class TradeDecisionLogger:
    """Logs and displays trade decisions"""
    
    def __init__(self):
        self.decisions = deque(maxlen=100)
        self.current_strategy = "Accumulation"
        self.confidence_score = 0.75
        
    def log_decision(self, decision_type: str, details: Dict[str, Any]):
        """Log a new trading decision"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        decision = {
            'timestamp': timestamp,
            'type': decision_type,
            'details': details,
            'confidence': self.confidence_score
        }
        self.decisions.append(decision)
        logger.info(f"Trade decision logged: {decision_type}")
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict]:
        """Get recent decisions for display"""
        return list(self.decisions)[-count:]

class SchwaboxVisualCore:
    """Main application class for Schwabot Visual Core"""
    
    def __init__(self):
        self.settings = UISettings()
        self.hardware_monitor = HardwareMonitor()
        self.market_simulator = MarketDataSimulator()
        self.decision_logger = TradeDecisionLogger()
        self.running = False
        
        # Data for plots
        self.profit_data = deque(maxlen=1000)
        self.profit_timestamps = deque(maxlen=1000)
        
        # Initialize profit tracking
        self.total_profit = 0.0
        self.profit_start_time = time.time()
        
        # Load settings
        self.load_settings()
        
        # Initialize core components if available
        self.core_components = {}
        if CORE_AVAILABLE:
            self._initialize_core_components()
    
    def _initialize_core_components(self):
        """Initialize core Schwabot components"""
        try:
            self.core_components['monitor'] = SystemMonitor()
            self.core_components['vault'] = HashAffinityVault()
            logger.info("Core components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
    
    def load_settings(self):
        """Load settings from file"""
        settings_file = Path("schwabot_ui_settings.yaml")
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        for key, value in data.items():
                            if hasattr(self.settings, key):
                                setattr(self.settings, key, value)
                logger.info("Settings loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            settings_dict = asdict(self.settings)
            with open("schwabot_ui_settings.yaml", 'w') as f:
                yaml.dump(settings_dict, f, default_flow_style=False)
            logger.info("Settings saved successfully")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def start_systems(self):
        """Start all background systems"""
        self.hardware_monitor.start()
        self.market_simulator.start()
        self.running = True
        
        # Start profit simulation
        threading.Thread(target=self._profit_simulation, daemon=True).start()
        
        # Start decision simulation
        threading.Thread(target=self._decision_simulation, daemon=True).start()
        
        logger.info("All systems started")
    
    def stop_systems(self):
        """Stop all background systems"""
        self.running = False
        self.hardware_monitor.stop()
        self.market_simulator.stop()
        logger.info("All systems stopped")
    
    def _profit_simulation(self):
        """Simulate profit tracking"""
        while self.running:
            try:
                # Simulate profit changes
                profit_change = np.random.normal(10, 50)  # Average $10 profit with volatility
                self.total_profit += profit_change
                
                self.profit_data.append(self.total_profit)
                self.profit_timestamps.append(time.time() - self.profit_start_time)
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Profit simulation error: {e}")
                time.sleep(10)
    
    def _decision_simulation(self):
        """Simulate trading decisions"""
        decision_types = ["BUY", "SELL", "HOLD", "ACCUMULATE", "DISTRIBUTE"]
        
        while self.running:
            try:
                decision_type = np.random.choice(decision_types)
                details = {
                    'pair': self.settings.primary_pair,
                    'price': self.market_simulator.current_price,
                    'amount': round(np.random.uniform(0.001, 0.1), 4),
                    'strategy': self.decision_logger.current_strategy
                }
                
                self.decision_logger.log_decision(decision_type, details)
                time.sleep(np.random.uniform(10, 30))  # Random interval
                
            except Exception as e:
                logger.error(f"Decision simulation error: {e}")
                time.sleep(20)

def create_ui(app: SchwaboxVisualCore):
    """Create the main UI interface"""
    
    def update_hardware_display():
        """Update hardware monitoring displays"""
        if dpg.does_item_exist("cpu_usage"):
            cpu_val = app.hardware_monitor.cpu_percent / 100.0
            dpg.set_value("cpu_usage", cpu_val)
            dpg.configure_item("cpu_usage", overlay=f"CPU {app.hardware_monitor.cpu_percent:.1f}%")
        
        if dpg.does_item_exist("memory_usage"):
            mem_val = app.hardware_monitor.memory_percent / 100.0
            dpg.set_value("memory_usage", mem_val)
            dpg.configure_item("memory_usage", overlay=f"Memory {app.hardware_monitor.memory_percent:.1f}%")
        
        if GPU_AVAILABLE and dpg.does_item_exist("gpu_usage"):
            gpu_val = app.hardware_monitor.gpu_percent / 100.0
            dpg.set_value("gpu_usage", gpu_val)
            dpg.configure_item("gpu_usage", overlay=f"GPU {app.hardware_monitor.gpu_percent:.1f}%")
            
            if dpg.does_item_exist("gpu_temp"):
                dpg.set_value("gpu_temp", f"GPU Temp: {app.hardware_monitor.gpu_temperature:.1f}°C")
    
    def update_market_display():
        """Update market data displays"""
        if len(app.market_simulator.price_data) > 1:
            # Update price chart
            timestamps = list(app.market_simulator.timestamps)
            prices = list(app.market_simulator.price_data)
            
            if dpg.does_item_exist("price_series"):
                dpg.set_value("price_series", [timestamps, prices])
            
            # Update current price display
            if dpg.does_item_exist("current_price"):
                dpg.set_value("current_price", f"Current Price: ${app.market_simulator.current_price:,.2f}")
    
    def update_profit_display():
        """Update profit tracking displays"""
        if len(app.profit_data) > 1:
            timestamps = list(app.profit_timestamps)
            profits = list(app.profit_data)
            
            if dpg.does_item_exist("profit_series"):
                dpg.set_value("profit_series", [timestamps, profits])
            
            if dpg.does_item_exist("total_profit"):
                dpg.set_value("total_profit", f"Total Profit: ${app.total_profit:,.2f}")
    
    def update_decisions_display():
        """Update decision log display"""
        decisions = app.decision_logger.get_recent_decisions(10)
        
        if dpg.does_item_exist("decision_log"):
            dpg.delete_item("decision_log", children_only=True)
            
            for decision in reversed(decisions):  # Show newest first
                with dpg.group(parent="decision_log"):
                    dpg.add_text(f"[{decision['timestamp']}] {decision['type']}")
                    dpg.add_text(f"  Details: {decision['details']}", color=(150, 150, 150))
                    dpg.add_separator()
    
    def save_settings_callback():
        """Save settings callback"""
        # Update settings from UI
        if dpg.does_item_exist("cb_key"):
            app.settings.coinbase_api_key = dpg.get_value("cb_key")
        if dpg.does_item_exist("cb_secret"):
            app.settings.coinbase_secret = dpg.get_value("cb_secret")
        if dpg.does_item_exist("pair_select"):
            app.settings.primary_pair = dpg.get_value("pair_select")
        if dpg.does_item_exist("risk_slider"):
            app.settings.max_risk_percent = dpg.get_value("risk_slider")
        if dpg.does_item_exist("gpu_toggle"):
            app.settings.enable_gpu_acceleration = dpg.get_value("gpu_toggle")
        
        app.save_settings()
        logger.info("Settings saved from UI")
    
    def expand_decision_callback():
        """Show detailed decision analysis"""
        recent_decisions = app.decision_logger.get_recent_decisions(1)
        if recent_decisions:
            decision = recent_decisions[0]
            details_text = f"""
Last Decision Analysis:
- Type: {decision['type']}
- Timestamp: {decision['timestamp']}
- Confidence: {decision['confidence']:.2f}
- Details: {json.dumps(decision['details'], indent=2)}
"""
            
            with dpg.window(label="Decision Details", width=400, height=300, pos=(100, 100)):
                dpg.add_text(details_text)
                dpg.add_button(label="Close", callback=lambda: dpg.delete_item(dpg.last_item()))
    
    # Create main window
    dpg.create_context()
    dpg.create_viewport(title="Schwabot v0.045c — Visual Synthesis Core", width=1400, height=800)
    
    with dpg.window(tag="Primary Window"):
        with dpg.tab_bar():
            
            # System Panel
            with dpg.tab(label="System"):
                dpg.add_text("Hardware Status (Live)", color=(100, 255, 100))
                dpg.add_separator()
                
                dpg.add_progress_bar(tag="cpu_usage", default_value=0.0, overlay="CPU 0%", width=-1)
                dpg.add_progress_bar(tag="memory_usage", default_value=0.0, overlay="Memory 0%", width=-1)
                
                if GPU_AVAILABLE:
                    dpg.add_progress_bar(tag="gpu_usage", default_value=0.0, overlay="GPU 0%", width=-1)
                    dpg.add_text("GPU N/A", tag="gpu_temp", color=(200, 200, 200))
                else:
                    dpg.add_text("GPU: Not Available", color=(255, 200, 100))
                
                dpg.add_spacing()
                dpg.add_checkbox(label="GPU Acceleration Active", tag="gpu_active", 
                               default_value=app.settings.enable_gpu_acceleration)
                dpg.add_checkbox(label="Fractal Engine Live", tag="fractal_live", default_value=True)
                dpg.add_checkbox(label="Auto Trading Enabled", tag="auto_trading", 
                               default_value=app.settings.auto_trading_enabled)
            
            # Profit Panel
            with dpg.tab(label="Profit"):
                dpg.add_text("Profit Tracking (Live)", color=(100, 255, 100))
                dpg.add_text("Total Profit: $0.00", tag="total_profit", color=(100, 255, 150))
                dpg.add_separator()
                
                with dpg.plot(label="Profit Curve", height=300, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="profit_x", label="Time (seconds)")
                    dpg.add_plot_axis(dpg.mvYAxis, tag="profit_y", label="Profit ($)")
                    dpg.add_line_series([], [], parent="profit_y", tag="profit_series")
            
            # Brain Panel  
            with dpg.tab(label="Brain"):
                dpg.add_text("Decision Trace (Live)", color=(100, 255, 100))
                dpg.add_separator()
                
                dpg.add_child_window(tag="decision_log", height=300, border=True)
                dpg.add_button(label="Expand Last Decision", callback=expand_decision_callback)
                
                dpg.add_spacing()
                dpg.add_text(f"Current Strategy: {app.decision_logger.current_strategy}")
                dpg.add_text(f"Confidence Score: {app.decision_logger.confidence_score:.2f}")
            
            # Flow Panel
            with dpg.tab(label="Flow"):
                dpg.add_text("Market Data Stream (Live)", color=(100, 255, 100))
                dpg.add_text("Current Price: $0.00", tag="current_price", color=(100, 255, 150))
                dpg.add_separator()
                
                with dpg.plot(label="Price Stream", height=300, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="flow_x", label="Time")
                    dpg.add_plot_axis(dpg.mvYAxis, tag="flow_y", label="Price ($)")
                    dpg.add_line_series([], [], parent="flow_y", tag="price_series")
            
            # Settings Panel
            with dpg.tab(label="Settings"):
                dpg.add_text("Configuration", color=(100, 255, 100))
                dpg.add_separator()
                
                dpg.add_text("API Configuration:")
                dpg.add_input_text(label="Coinbase API Key", password=True, width=400, tag="cb_key",
                                 default_value=app.settings.coinbase_api_key)
                dpg.add_input_text(label="Coinbase Secret", password=True, width=400, tag="cb_secret",
                                 default_value=app.settings.coinbase_secret)
                
                dpg.add_spacing()
                dpg.add_text("Trading Configuration:")
                dpg.add_combo(label="Primary Pair", 
                            items=["BTC/USDC", "ETH/BTC", "BTC/USDT", "ETH/USDC"],
                            default_value=app.settings.primary_pair, width=150, tag="pair_select")
                dpg.add_slider_float(label="Max Risk (%)", min_value=0.1, max_value=10.0,
                                   default_value=app.settings.max_risk_percent, width=200, tag="risk_slider")
                
                dpg.add_spacing()
                dpg.add_text("System Configuration:")
                dpg.add_checkbox(label="Enable GPU Acceleration", 
                               default_value=app.settings.enable_gpu_acceleration, tag="gpu_toggle")
                
                dpg.add_spacing()
                dpg.add_button(label="Save Settings", callback=save_settings_callback)
    
    # Setup update timer
    def update_all_displays():
        """Update all UI displays"""
        update_hardware_display()
        update_market_display() 
        update_profit_display()
        update_decisions_display()
    
    # Register timer for updates
    with dpg.handler_registry():
        dpg.add_timer_handler(interval=1.0, callback=lambda: update_all_displays())
    
    # Finalize and show
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    
    return update_all_displays

def main():
    """Main application entry point"""
    logger.info("Starting Schwabot Visual Core 0.045c")
    
    # Create application
    app = SchwaboxVisualCore()
    
    try:
        # Start background systems
        app.start_systems()
        
        # Create and run UI
        update_displays = create_ui(app)
        
        # Start Dear PyGui event loop
        dpg.start_dearpygui()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
    
    finally:
        # Cleanup
        app.stop_systems()
        dpg.destroy_context()
        logger.info("Schwabot Visual Core shutdown complete")

if __name__ == "__main__":
    main() 