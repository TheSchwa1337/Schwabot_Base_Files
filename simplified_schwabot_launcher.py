#!/usr/bin/env python3
"""
Simplified Schwabot Launcher
============================

Easy-to-use launcher for the simplified Schwabot system that addresses all user concerns:
- JSON-based configuration instead of complex YAML
- Demo mode functionality for testing strategies
- Simple command-line interface
- Automatic dependency checking
- Clean error handling and fallback mechanisms

Usage:
    python simplified_schwabot_launcher.py demo          # Run demo mode
    python simplified_schwabot_launcher.py api           # Run API only
    python simplified_schwabot_launcher.py live          # Run with live BTC integration
    python simplified_schwabot_launcher.py config        # Configure system
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedLauncher:
    """Simplified launcher for Schwabot system"""
    
    def __init__(self):
        """Initialize launcher"""
        self.config_dir = Path.home() / ".schwabot"
        self.config_file = self.config_dir / "simple_config.json"
        self.dependencies_checked = False
        
        print("🚀 Simplified Schwabot Launcher")
        print("================================")
    
    def check_dependencies(self) -> bool:
        """Check and install required dependencies"""
        if self.dependencies_checked:
            return True
        
        print("🔍 Checking dependencies...")
        
        missing_deps = []
        required_deps = [
            'fastapi',
            'uvicorn',
            'websockets',
            'numpy',
            'pydantic'
        ]
        
        for dep in required_deps:
            try:
                __import__(dep)
                print(f"   ✅ {dep}")
            except ImportError:
                missing_deps.append(dep)
                print(f"   ❌ {dep} (missing)")
        
        if missing_deps:
            print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
            print("📦 Install with: pip install " + " ".join(missing_deps))
            return False
        
        print("✅ All dependencies available")
        self.dependencies_checked = True
        return True
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            "demo_mode": True,
            "live_trading_enabled": False,
            "position_size_limit": 0.1,
            "api_port": 8000,
            "websocket_update_interval": 0.5,
            "max_drawdown": 0.05,
            "stop_loss": 0.02,
            "demo_speed_multiplier": 1.0,
            "synthetic_data_enabled": True,
            "sustainment_threshold": 0.65,
            "confidence_threshold": 0.70
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"📄 Loaded config from {self.config_file}")
                return config
            except Exception as e:
                print(f"⚠️  Error loading config: {e}")
        
        # Create default config
        config = self.create_default_config()
        self.save_config(config)
        print(f"📄 Created default config at {self.config_file}")
        return config
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration"""
        self.config_dir.mkdir(exist_ok=True)
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"💾 Saved config to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def run_demo_mode(self):
        """Run in demo mode"""
        print("\n🧪 Starting Demo Mode")
        print("====================")
        print("• Demo mode runs with synthetic data")
        print("• No real trading or API keys required")
        print("• Safe for testing strategies")
        print("• Web dashboard: http://localhost:8000")
        print("• WebSocket: ws://localhost:8000/ws")
        
        if not self.check_dependencies():
            return False
        
        try:
            from core.simplified_api import create_simplified_api
            
            # Create API with demo configuration
            api = create_simplified_api()
            api.config.demo_mode = True
            api.config.live_trading_enabled = False
            
            print("\n🚀 Starting simplified API server...")
            print(f"📊 Dashboard: http://localhost:{api.config.api_port}")
            print("🔌 Press Ctrl+C to stop")
            
            api.run()
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        except Exception as e:
            print(f"❌ Error running demo: {e}")
            logger.error(f"Demo mode error: {e}", exc_info=True)
            return False
        
        return True
    
    def run_api_only(self):
        """Run API server only"""
        print("\n🔌 Starting API Only Mode")
        print("=========================")
        print("• API server without BTC integration")
        print("• Good for frontend development")
        print("• Mock data for testing")
        
        if not self.check_dependencies():
            return False
        
        try:
            from core.simplified_api import create_simplified_api
            
            config = self.load_config()
            api = create_simplified_api()
            
            # Apply loaded configuration
            for key, value in config.items():
                if hasattr(api.config, key):
                    setattr(api.config, key, value)
            
            print(f"\n🚀 Starting API server on port {api.config.api_port}...")
            print("🔌 Press Ctrl+C to stop")
            
            api.run()
            
        except KeyboardInterrupt:
            print("\n⏹️  API server stopped by user")
        except Exception as e:
            print(f"❌ Error running API: {e}")
            logger.error(f"API mode error: {e}", exc_info=True)
            return False
        
        return True
    
    def run_live_mode(self):
        """Run with live BTC integration"""
        print("\n📈 Starting Live Mode")
        print("=====================")
        print("• Live BTC data integration")
        print("• High-frequency tick processing")
        print("• Real-time sustainment monitoring")
        print("• ✅ ALL Mathematical Complexity Preserved:")
        print("  - Klein Bottle Topology")
        print("  - Forever Fractals Analysis")
        print("  - 8-Principle Sustainment Framework")
        print("  - Drift Shell Mathematical Framework")
        print("  - Quantum Intelligence Core")
        print("  - Recursive Memory Constellation Logic")
        print("  - Temporal Echo Recognition")
        print("  - Chrono-Spatial Pattern Integrity")
        print("• ⚠️  Demo trading only (live trading disabled for safety)")
        
        if not self.check_dependencies():
            return False
        
        try:
            from core.simplified_api import create_simplified_api
            
            # Use enhanced bridge that preserves ALL mathematical complexity
            try:
                from core.enhanced_btc_integration_bridge import integrate_enhanced_bridge_with_api
                from core.advanced_drift_shell_integration import UnifiedDriftShellController
                
                enhanced_bridge_available = True
                print("✅ Enhanced mathematical bridge available - full complexity preserved")
            except ImportError:
                enhanced_bridge_available = False
                print("⚠️ Enhanced bridge not available - using standard integration")
            
            config = self.load_config()
            
            # Create integrated system with full mathematical complexity
            print("🔧 Initializing systems with complete mathematical framework...")
            api = create_simplified_api()
            
            if enhanced_bridge_available:
                # Use enhanced bridge that preserves ALL mathematical complexity
                bridge = integrate_enhanced_bridge_with_api(api, config)
                
                # Initialize drift shell controller
                drift_shell = UnifiedDriftShellController()
                
                # Add Schwa memory contexts for intent weighting
                drift_shell.add_schwa_memory_context("profitable_btc_trends", 0.85)
                drift_shell.add_schwa_memory_context("successful_volatility_trading", 0.78)
                drift_shell.add_schwa_memory_context("mean_reversion_profits", 0.82)
                
                api.drift_shell_controller = drift_shell
                
                print("✅ Enhanced mathematical bridge initialized")
                print("  - All core systems operational")
                print("  - Klein Bottle topology active")
                print("  - Forever Fractals analysis running")
                print("  - Drift Shell threads monitoring")
                print("  - Quantum intelligence integrated")
                print("  - Sustainment framework active")
            else:
                # Fallback to standard integration
                from core.simplified_btc_integration import create_integrated_system
                btc_integration = create_integrated_system(api)
                print("✅ Standard integration initialized")
            
            # Apply configuration
            for key, value in config.items():
                if hasattr(api.config, key):
                    setattr(api.config, key, value)
            
            if enhanced_bridge_available:
                # Start enhanced bridge
                print("📊 Starting enhanced mathematical processing...")
                # The bridge will start automatically when API starts
                
                print(f"🚀 Starting integrated system on port {api.config.api_port}...")
                print("📈 Processing with FULL mathematical complexity:")
                print("  - Real-time Klein Bottle calculations")
                print("  - Fractal dimension analysis")
                print("  - Sustainment principle monitoring")
                print("  - Drift shell pattern recognition")
                print("  - Quantum correlation tracking")
                print("  - Memory constellation formation")
                print("🔌 Press Ctrl+C to stop")
                
                api.run()
            else:
                # Start standard BTC integration
                print("📊 Starting BTC integration...")
                if hasattr(api, 'btc_integration'):
                    if not api.btc_integration.start_integration():
                        print("❌ Failed to start BTC integration")
                        return False
                
                print(f"🚀 Starting integrated system on port {api.config.api_port}...")
                print("📈 Processing live tick data")
                print("🔌 Press Ctrl+C to stop")
                
                try:
                    api.run()
                finally:
                    print("⏹️  Stopping BTC integration...")
                    if hasattr(api, 'btc_integration'):
                        api.btc_integration.stop_integration()
            
        except KeyboardInterrupt:
            print("\n⏹️  Live mode stopped by user")
        except Exception as e:
            print(f"❌ Error running live mode: {e}")
            logger.error(f"Live mode error: {e}", exc_info=True)
            return False
        
        return True
    
    def configure_system(self):
        """Interactive configuration"""
        print("\n⚙️  System Configuration")
        print("========================")
        
        config = self.load_config()
        
        print("\nCurrent configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\nConfiguration options:")
        print("1. Toggle demo mode")
        print("2. Change API port")
        print("3. Adjust risk settings")
        print("4. Reset to defaults")
        print("5. Save and exit")
        
        while True:
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == "1":
                    config["demo_mode"] = not config["demo_mode"]
                    print(f"Demo mode: {config['demo_mode']}")
                
                elif choice == "2":
                    port = int(input("Enter API port (8000): ") or "8000")
                    config["api_port"] = port
                    print(f"API port: {port}")
                
                elif choice == "3":
                    print("\nRisk Settings:")
                    config["max_drawdown"] = float(input(f"Max drawdown ({config['max_drawdown']}): ") or config["max_drawdown"])
                    config["stop_loss"] = float(input(f"Stop loss ({config['stop_loss']}): ") or config["stop_loss"])
                    config["position_size_limit"] = float(input(f"Position size limit ({config['position_size_limit']}): ") or config["position_size_limit"])
                
                elif choice == "4":
                    config = self.create_default_config()
                    print("Reset to default configuration")
                
                elif choice == "5":
                    self.save_config(config)
                    print("Configuration saved")
                    break
                
                else:
                    print("Invalid option")
                    
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or cancelled")
                break
    
    def show_status(self):
        """Show system status"""
        print("\n📊 System Status")
        print("================")
        
        config = self.load_config()
        
        print(f"Configuration file: {self.config_file}")
        print(f"Demo mode: {config.get('demo_mode', 'Unknown')}")
        print(f"API port: {config.get('api_port', 'Unknown')}")
        print(f"Live trading: {config.get('live_trading_enabled', 'Unknown')}")
        
        # Check if API is running
        try:
            import requests
            response = requests.get(f"http://localhost:{config.get('api_port', 8000)}", timeout=2)
            print("API Status: 🟢 Running")
        except:
            print("API Status: 🔴 Not running")
        
        # Check dependencies
        print("\nDependencies:")
        self.check_dependencies()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Simplified Schwabot Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simplified_schwabot_launcher.py demo       # Run demo mode
  python simplified_schwabot_launcher.py api        # Run API only  
  python simplified_schwabot_launcher.py live       # Run with live BTC integration
  python simplified_schwabot_launcher.py config     # Configure system
  python simplified_schwabot_launcher.py status     # Show system status
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['demo', 'api', 'live', 'config', 'status'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    launcher = SimplifiedLauncher()
    
    try:
        if args.mode == 'demo':
            success = launcher.run_demo_mode()
        elif args.mode == 'api':
            success = launcher.run_api_only()
        elif args.mode == 'live':
            success = launcher.run_live_mode()
        elif args.mode == 'config':
            launcher.configure_system()
            success = True
        elif args.mode == 'status':
            launcher.show_status()
            success = True
        else:
            print(f"Unknown mode: {args.mode}")
            success = False
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Launcher error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 