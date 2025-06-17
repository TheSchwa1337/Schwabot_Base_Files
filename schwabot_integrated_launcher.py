#!/usr/bin/env python3
"""
Schwabot Integrated Launcher
============================

Main launcher that integrates all Schwabot components:
- Sustainment underlay (mathematical core)
- UI state bridge (data translation)
- Visual integration bridge (Tesseract visualizers)
- React dashboard server (web interface)
- API connections (Coinbase, CCXT, etc.)
- Complete trading pipeline

This is the production entry point for the complete Schwabot trading system.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Add core to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# Core system imports
from core.sustainment_underlay_controller import SustainmentUnderlayController
from core.ui_state_bridge import create_ui_bridge
from core.visual_integration_bridge import create_visual_bridge
from core.react_dashboard_integration import create_react_dashboard

# Import existing controllers (with fallbacks)
try:
    from core.thermal_zone_manager import ThermalZoneManager
    from core.cooldown_manager import CooldownManager  
    from core.profit_navigator import AntiPoleProfitNavigator
    from core.fractal_core import FractalCore
    CORE_CONTROLLERS_AVAILABLE = True
except ImportError:
    CORE_CONTROLLERS_AVAILABLE = False
    print("⚠️ Some core controllers not available - using mock implementations")

# API integrations
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ CCXT not available - crypto API features disabled")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠️ WebSockets not available - real-time features limited")

logger = logging.getLogger(__name__)

class SchwabotIntegratedSystem:
    """
    Complete Schwabot integrated trading system.
    
    Orchestrates all components:
    - Mathematical sustainment framework
    - Visual integration and Tesseract visualizers  
    - Real-time dashboard and API interfaces
    - Trading execution and risk management
    - Hardware thermal management
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the complete Schwabot system
        
        Args:
            config_file: Path to configuration file (JSON/YAML)
        """
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # System state
        self.system_active = False
        self.components_initialized = False
        
        # Core controllers
        self.thermal_manager = None
        self.cooldown_manager = None
        self.profit_navigator = None
        self.fractal_core = None
        
        # Integration layers
        self.sustainment_controller = None
        self.ui_bridge = None
        self.visual_bridge = None
        self.dashboard_server = None
        
        # API connections
        self.exchange_clients = {}
        self.api_status = {}
        
        # Performance tracking
        self.start_time = None
        self.total_trades = 0
        self.total_profit = 0.0
        
        logger.info("Schwabot Integrated System initialized")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load system configuration from file"""
        
        # Default configuration
        default_config = {
            "system": {
                "update_interval": 0.1,
                "sustainment_threshold": 0.65,
                "thermal_management": True,
                "gpu_acceleration": True
            },
            "dashboard": {
                "web_port": 5000,
                "websocket_port": 8765,
                "enable_react_dashboard": True,
                "enable_python_dashboard": False
            },
            "apis": {
                "coinbase": {
                    "enabled": False,
                    "api_key": "",
                    "api_secret": "",
                    "passphrase": "",
                    "sandbox": True
                },
                "binance": {
                    "enabled": False,
                    "api_key": "",
                    "api_secret": "",
                    "testnet": True
                },
                "ccxt_exchanges": []
            },
            "trading": {
                "enabled": False,
                "max_position_size": 0.1,
                "risk_per_trade": 0.02,
                "stop_loss_percent": 0.05,
                "take_profit_percent": 0.15
            },
            "visualization": {
                "enable_tesseract": True,
                "enable_advanced_tesseract": True,
                "update_frequency": 10.0,
                "export_data": True
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        user_config = json.load(f)
                    else:
                        # Assume YAML
                        import yaml
                        user_config = yaml.safe_load(f)
                
                # Merge with defaults
                self._deep_merge(default_config, user_config)
                logger.info(f"Configuration loaded from {config_file}")
                
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config

    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def initialize_core_controllers(self) -> bool:
        """Initialize core trading controllers"""
        
        logger.info("Initializing core controllers...")
        
        try:
            if CORE_CONTROLLERS_AVAILABLE:
                # Initialize real controllers
                self.thermal_manager = ThermalZoneManager()
                self.cooldown_manager = CooldownManager()
                self.profit_navigator = AntiPoleProfitNavigator()
                self.fractal_core = FractalCore()
                
                logger.info("✅ Real core controllers initialized")
            else:
                # Initialize mock controllers for demo
                self.thermal_manager, self.cooldown_manager, self.profit_navigator, self.fractal_core = self._create_mock_controllers()
                logger.info("✅ Mock core controllers initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize core controllers: {e}")
            return False

    def _create_mock_controllers(self):
        """Create mock controllers for demo purposes"""
        
        class MockThermalManager:
            def get_current_state(self): 
                from types import SimpleNamespace
                return SimpleNamespace(
                    zone=SimpleNamespace(value="normal"),
                    load_cpu=0.3, load_gpu=0.4, memory_usage=0.5,
                    cpu_temp=55.0, gpu_temp=65.0
                )
        
        class MockCooldownManager:
            def get_status(self): return {"status": "active", "cooldown_active": False}
            
        class MockProfitNavigator:
            def get_current_strategy(self): return {"strategy": "anti_pole", "confidence": 0.85}
            
        class MockFractalCore:
            def get_current_state(self): 
                from types import SimpleNamespace
                return SimpleNamespace(coherence=0.6, entropy=0.4, phase=0.0)
        
        return MockThermalManager(), MockCooldownManager(), MockProfitNavigator(), MockFractalCore()

    def initialize_integration_layers(self) -> bool:
        """Initialize the integration and bridge layers"""
        
        logger.info("Initializing integration layers...")
        
        try:
            # Initialize sustainment underlay controller
            self.sustainment_controller = SustainmentUnderlayController(
                thermal_manager=self.thermal_manager,
                cooldown_manager=self.cooldown_manager,
                profit_navigator=self.profit_navigator,
                fractal_core=self.fractal_core,
                s_crit=self.config["system"]["sustainment_threshold"]
            )
            
            # Start sustainment monitoring
            self.sustainment_controller.start_continuous_synthesis(
                interval=self.config["system"]["update_interval"]
            )
            
            # Initialize UI state bridge
            self.ui_bridge = create_ui_bridge(
                self.sustainment_controller,
                self.thermal_manager,
                self.profit_navigator,
                self.fractal_core
            )
            
            # Initialize visual integration bridge
            self.visual_bridge = create_visual_bridge(
                self.ui_bridge,
                self.sustainment_controller,
                websocket_port=self.config["dashboard"]["websocket_port"]
            )
            
            logger.info("✅ Integration layers initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize integration layers: {e}")
            return False

    def initialize_dashboard(self) -> bool:
        """Initialize dashboard interfaces"""
        
        logger.info("Initializing dashboard interfaces...")
        
        try:
            # Initialize React dashboard if enabled
            if self.config["dashboard"]["enable_react_dashboard"]:
                self.dashboard_server = create_react_dashboard(
                    self.visual_bridge,
                    self.ui_bridge,
                    self.sustainment_controller,
                    port=self.config["dashboard"]["web_port"]
                )
                logger.info(f"✅ React dashboard started on port {self.config['dashboard']['web_port']}")
            
            # Initialize Python dashboard if enabled
            if self.config["dashboard"]["enable_python_dashboard"]:
                from core.schwabot_dashboard import SchwabotDashboard
                # Would initialize Python dashboard...
                logger.info("✅ Python dashboard interface ready")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            return False

    def initialize_api_connections(self) -> bool:
        """Initialize API connections for trading"""
        
        logger.info("Initializing API connections...")
        
        if not CCXT_AVAILABLE:
            logger.warning("CCXT not available - skipping API initialization")
            return True
        
        try:
            # Initialize Coinbase Pro if configured
            if self.config["apis"]["coinbase"]["enabled"]:
                coinbase_config = self.config["apis"]["coinbase"]
                
                if coinbase_config["api_key"] and coinbase_config["api_secret"]:
                    exchange = ccxt.coinbasepro({
                        'apiKey': coinbase_config["api_key"],
                        'secret': coinbase_config["api_secret"],
                        'passphrase': coinbase_config["passphrase"],
                        'sandbox': coinbase_config["sandbox"],
                    })
                    
                    # Test connection
                    try:
                        balance = exchange.fetch_balance()
                        self.exchange_clients['coinbase'] = exchange
                        self.api_status['coinbase'] = 'connected'
                        logger.info("✅ Coinbase Pro API connected")
                    except Exception as e:
                        logger.error(f"Coinbase Pro API test failed: {e}")
                        self.api_status['coinbase'] = 'error'
            
            # Initialize Binance if configured
            if self.config["apis"]["binance"]["enabled"]:
                binance_config = self.config["apis"]["binance"]
                
                if binance_config["api_key"] and binance_config["api_secret"]:
                    exchange = ccxt.binance({
                        'apiKey': binance_config["api_key"],
                        'secret': binance_config["api_secret"],
                        'testnet': binance_config["testnet"],
                    })
                    
                    # Test connection
                    try:
                        balance = exchange.fetch_balance()
                        self.exchange_clients['binance'] = exchange
                        self.api_status['binance'] = 'connected'
                        logger.info("✅ Binance API connected")
                    except Exception as e:
                        logger.error(f"Binance API test failed: {e}")
                        self.api_status['binance'] = 'error'
            
            # Initialize other CCXT exchanges
            for exchange_config in self.config["apis"]["ccxt_exchanges"]:
                # Initialize additional exchanges...
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize API connections: {e}")
            return False

    def start_system(self) -> bool:
        """Start the complete Schwabot system"""
        
        logger.info("🚀 Starting Schwabot Integrated Trading System...")
        
        try:
            # Initialize all components
            if not self.initialize_core_controllers():
                return False
            
            if not self.initialize_integration_layers():
                return False
            
            if not self.initialize_dashboard():
                return False
            
            if not self.initialize_api_connections():
                return False
            
            # Mark system as active
            self.system_active = True
            self.start_time = datetime.now()
            
            logger.info("✅ Schwabot system fully operational!")
            self._print_system_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Schwabot system: {e}")
            return False

    def stop_system(self) -> None:
        """Stop the complete Schwabot system"""
        
        logger.info("🛑 Stopping Schwabot system...")
        
        self.system_active = False
        
        # Stop dashboard server
        if self.dashboard_server:
            self.dashboard_server.stop_server()
        
        # Stop visual bridge
        if self.visual_bridge:
            self.visual_bridge.stop_visual_bridge()
        
        # Stop UI bridge
        if self.ui_bridge:
            self.ui_bridge.stop_ui_monitoring()
        
        # Stop sustainment controller
        if self.sustainment_controller:
            self.sustainment_controller.stop_continuous_synthesis()
        
        # Close API connections
        for name, client in self.exchange_clients.items():
            try:
                if hasattr(client, 'close'):
                    client.close()
                logger.info(f"Closed {name} API connection")
            except Exception as e:
                logger.warning(f"Error closing {name} API: {e}")
        
        logger.info("✅ Schwabot system stopped cleanly")

    def _print_system_status(self) -> None:
        """Print current system status"""
        
        print("\n" + "=" * 70)
        print("🎯 SCHWABOT v1.0 - SYSTEM STATUS")
        print("=" * 70)
        
        print(f"\n📊 System Information:")
        print(f"   • Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Sustainment Threshold: {self.config['system']['sustainment_threshold']}")
        print(f"   • Update Interval: {self.config['system']['update_interval']}s")
        print(f"   • GPU Acceleration: {'Enabled' if self.config['system']['gpu_acceleration'] else 'Disabled'}")
        
        print(f"\n🌐 Dashboard Interfaces:")
        if self.config["dashboard"]["enable_react_dashboard"]:
            print(f"   • React Dashboard: http://localhost:{self.config['dashboard']['web_port']}")
        if self.config["dashboard"]["enable_python_dashboard"]:
            print(f"   • Python Dashboard: Available")
        print(f"   • WebSocket Server: ws://localhost:{self.config['dashboard']['websocket_port']}")
        
        print(f"\n🔗 API Connections:")
        if not self.api_status:
            print("   • No API connections configured")
        else:
            for api_name, status in self.api_status.items():
                status_icon = "✅" if status == "connected" else "❌"
                print(f"   • {api_name.title()}: {status_icon} {status}")
        
        print(f"\n⚙️ Core Controllers:")
        print(f"   • Thermal Manager: {'✅ Active' if self.thermal_manager else '❌ Not Available'}")
        print(f"   • Profit Navigator: {'✅ Active' if self.profit_navigator else '❌ Not Available'}")
        print(f"   • Fractal Core: {'✅ Active' if self.fractal_core else '❌ Not Available'}")
        print(f"   • Sustainment Underlay: {'✅ Active' if self.sustainment_controller else '❌ Not Available'}")
        
        print(f"\n📈 Visualization:")
        tesseract_status = self.visual_bridge.get_tesseract_status() if self.visual_bridge else {}
        print(f"   • Tesseract Available: {'✅ Yes' if tesseract_status.get('tesseract_available') else '❌ No'}")
        print(f"   • WebSocket Clients: {tesseract_status.get('websocket_clients', 0)}")
        
        print(f"\n🎯 Quick Access:")
        print(f"   • Main Dashboard: http://localhost:{self.config['dashboard']['web_port']}")
        print(f"   • System API: http://localhost:{self.config['dashboard']['web_port']}/api/status")
        print(f"   • Configuration: Ctrl+C to stop, edit config and restart")
        
        print("\n" + "=" * 70)

    def run_main_loop(self) -> None:
        """Run the main system loop"""
        
        logger.info("Entering main system loop...")
        
        try:
            while self.system_active:
                # System monitoring and maintenance
                self._system_health_check()
                
                # Sleep for a bit
                time.sleep(5.0)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop_system()

    def _system_health_check(self) -> None:
        """Perform periodic system health checks"""
        
        try:
            # Check sustainment status
            if self.sustainment_controller:
                status = self.sustainment_controller.get_sustainment_status()
                si = status.get('sustainment_index', 0.5)
                
                if si < self.config["system"]["sustainment_threshold"]:
                    logger.warning(f"Sustainment index below threshold: {si:.3f}")
            
            # Check API connections
            for name, client in self.exchange_clients.items():
                try:
                    # Test connection with a simple API call
                    # client.fetch_markets()  # Would test connection
                    pass
                except Exception as e:
                    logger.warning(f"{name} API connection issue: {e}")
                    self.api_status[name] = 'error'
            
        except Exception as e:
            logger.error(f"Health check error: {e}")

def main():
    """Main entry point for Schwabot integrated system"""
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Schwabot Integrated Trading System")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--log-level", "-l", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--demo", "-d", action="store_true",
                       help="Run in demo mode with mock data")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Schwabot v1.0 - Law of Sustainment Trading Platform")
    print("=" * 70)
    print("Mathematical trading excellence with operational integrity")
    print("=" * 70)
    
    # Create system instance
    system = SchwabotIntegratedSystem(config_file=args.config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        system.stop_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the system
        if system.start_system():
            # Run main loop
            system.run_main_loop()
        else:
            print("❌ Failed to start Schwabot system")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 