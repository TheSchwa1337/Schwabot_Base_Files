#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoboldCPP Complete Trading System Launcher
=========================================

Launches the complete unified mathematical trading system with:
1. KoboldCPP AI integration
2. Mathematical components (tensor memory, strategy mapper, etc.)
3. Secure web interface
4. Real-time trading capabilities
5. Multi-cryptocurrency support
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('koboldcpp_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class KoboldCPPCompleteSystemLauncher:
    """Complete system launcher for KoboldCPP trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the system launcher."""
        self.config = self._load_config(config_path)
        self.running = False
        self.components = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 KoboldCPP Complete System Launcher initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"❌ Failed to load config from {config_path}: {e}")
        
        # Default configuration
        config = {
            "system": {
                "name": "KoboldCPP Unified Mathematical Trading System",
                "version": "1.0.0",
                "debug": False
            },
            "koboldcpp": {
                "path": "koboldcpp",
                "port": 5001,
                "model_path": "",
                "auto_start": True,
                "hardware_detection": True
            },
            "web_interface": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False,
                "ssl_enabled": False,
                "session_timeout": 3600
            },
            "trading_system": {
                "flask_port": 5002,
                "enable_multi_bot": True,
                "enable_consensus": True,
                "enable_risk_management": True
            },
            "mathematical_components": {
                "tensor_memory": {
                    "enabled": True,
                    "max_memory_size": 1000,
                    "learning_rate": 0.01
                },
                "strategy_mapper": {
                    "enabled": True,
                    "randomization_enabled": True,
                    "bit_phase_support": [4, 8, 16, 32, 42]
                },
                "visual_engine": {
                    "enabled": True,
                    "pattern_detection": True
                }
            },
            "cryptocurrencies": {
                "supported": ["BTC", "ETH", "XRP", "SOL", "USDC"],
                "default_pairs": ["BTC/USDC", "ETH/USDC", "XRP/USDC", "SOL/USDC"],
                "bit_phase_default": 8
            },
            "security": {
                "session_timeout": 3600,
                "max_login_attempts": 3,
                "password_hash_rounds": 12
            },
            "performance": {
                "max_concurrent_requests": 10,
                "request_timeout": 30,
                "health_check_interval": 60
            }
        }
        
        logger.info("✅ Using default configuration")
        return config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        asyncio.create_task(self.shutdown())
    
    async def initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("🔧 Initializing system components...")
            
            # Import components
            from core.koboldcpp_integration import KoboldCPPIntegration
            from core.unified_mathematical_trading_system import UnifiedMathematicalTradingSystem
            from core.tensor_weight_memory import TensorWeightMemory
            from core.strategy_mapper import StrategyMapper
            from core.visual_decision_engine import VisualDecisionEngine
            from core.soulprint_registry import SoulprintRegistry
            from core.cascade_memory_architecture import CascadeMemoryArchitecture
            from gui.koboldcpp_web_interface import KoboldCPPWebInterface
            
            # Initialize KoboldCPP integration
            logger.info("🧠 Initializing KoboldCPP integration...")
            self.components['kobold_integration'] = KoboldCPPIntegration(
                kobold_path=self.config['koboldcpp']['path'],
                model_path=self.config['koboldcpp']['model_path'],
                port=self.config['koboldcpp']['port']
            )
            
            # Initialize mathematical components
            logger.info("⚛️ Initializing mathematical components...")
            self.components['tensor_memory'] = TensorWeightMemory(
                self.config['mathematical_components']['tensor_memory']
            )
            self.components['strategy_mapper'] = StrategyMapper(
                self.config['mathematical_components']['strategy_mapper']
            )
            self.components['visual_engine'] = VisualDecisionEngine(
                self.config['mathematical_components']['visual_engine']
            )
            self.components['soulprint_registry'] = SoulprintRegistry()
            self.components['cascade_memory'] = CascadeMemoryArchitecture()
            
            # Initialize trading system
            logger.info("📈 Initializing unified trading system...")
            self.components['trading_system'] = UnifiedMathematicalTradingSystem(
                self.config
            )
            
            # Initialize web interface
            logger.info("🌐 Initializing web interface...")
            self.components['web_interface'] = KoboldCPPWebInterface(
                self.config
            )
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    async def start_components(self):
        """Start all system components."""
        try:
            logger.info("🚀 Starting system components...")
            
            # Start KoboldCPP server
            if self.config['koboldcpp']['auto_start']:
                logger.info("🤖 Starting KoboldCPP server...")
                success = await self.components['kobold_integration'].start_kobold_server()
                if success:
                    logger.info("✅ KoboldCPP server started")
                else:
                    logger.warning("⚠️ KoboldCPP server failed to start, continuing without it")
            
            # Start tensor memory system
            logger.info("🧠 Starting tensor memory system...")
            self.components['tensor_memory'].start_memory_system()
            
            # Start trading system Flask server
            logger.info("📊 Starting trading system server...")
            self.components['trading_system'].start_flask_server(
                host='127.0.0.1',
                port=self.config['trading_system']['flask_port']
            )
            
            logger.info("✅ All components started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component startup failed: {e}")
            return False
    
    async def start_web_interface(self):
        """Start the web interface."""
        try:
            logger.info("🌐 Starting web interface...")
            
            # Start web interface in a separate thread
            import threading
            
            def run_web_interface():
                try:
                    self.components['web_interface'].start_server(
                        host=self.config['web_interface']['host'],
                        port=self.config['web_interface']['port'],
                        debug=self.config['web_interface']['debug']
                    )
                except Exception as e:
                    logger.error(f"❌ Web interface error: {e}")
            
            web_thread = threading.Thread(target=run_web_interface, daemon=True)
            web_thread.start()
            
            logger.info(f"✅ Web interface started on {self.config['web_interface']['host']}:{self.config['web_interface']['port']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Web interface startup failed: {e}")
            return False
    
    async def run_system(self):
        """Run the complete system."""
        try:
            logger.info("🎯 Starting KoboldCPP Complete Trading System...")
            
            # Initialize components
            if not await self.initialize_components():
                logger.error("❌ Failed to initialize components")
                return False
            
            # Start components
            if not await self.start_components():
                logger.error("❌ Failed to start components")
                return False
            
            # Start web interface
            if not await self.start_web_interface():
                logger.error("❌ Failed to start web interface")
                return False
            
            self.running = True
            
            # Display system information
            self._display_system_info()
            
            # Main loop
            logger.info("🔄 System running, press Ctrl+C to stop...")
            while self.running:
                await asyncio.sleep(1)
                
                # Health check
                if time.time() % self.config['performance']['health_check_interval'] == 0:
                    await self._health_check()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System runtime error: {e}")
            return False
    
    def _display_system_info(self):
        """Display system information."""
        print("\n" + "="*80)
        print("🤖 KOBOLDCPP UNIFIED MATHEMATICAL TRADING SYSTEM")
        print("="*80)
        print(f"📊 System Version: {self.config['system']['version']}")
        print(f"🌐 Web Interface: http://{self.config['web_interface']['host']}:{self.config['web_interface']['port']}")
        print(f"📈 Trading System: http://127.0.0.1:{self.config['trading_system']['flask_port']}")
        print(f"🤖 KoboldCPP Server: http://localhost:{self.config['koboldcpp']['port']}")
        print("\n🔐 Login Credentials:")
        print("   Username: admin | Password: admin123")
        print("   Username: trader | Password: trader123")
        print("\n💎 Supported Cryptocurrencies:")
        for crypto in self.config['cryptocurrencies']['supported']:
            print(f"   • {crypto}")
        print("\n⚛️ Mathematical Components:")
        print("   • Tensor Weight Memory System")
        print("   • Strategy Mapper with Bit Phase Logic")
        print("   • Visual Decision Engine")
        print("   • Soulprint Registry")
        print("   • Cascade Memory Architecture")
        print("\n🎯 Features:")
        print("   • Real-time AI-powered trading analysis")
        print("   • Multi-cryptocurrency support")
        print("   • 8-bit phase logic and strategy mapping")
        print("   • Secure web interface with real-time updates")
        print("   • Mathematical consensus integration")
        print("="*80)
        print("🚀 System is ready! Open your browser to access the dashboard.")
        print("="*80 + "\n")
    
    async def _health_check(self):
        """Perform system health check."""
        try:
            # Check KoboldCPP status
            kobold_status = self.components['kobold_integration'].get_statistics()
            
            # Check tensor memory status
            tensor_status = self.components['tensor_memory'].get_statistics()
            
            # Check trading system status
            trading_status = self.components['trading_system'].get_system_status()
            
            logger.info(f"💚 Health Check - KoboldCPP: {kobold_status.get('kobold_running', False)}, "
                       f"Tensor Memory: {tensor_status.get('enabled', False)}, "
                       f"Trading System: Active")
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    async def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            logger.info("🛑 Shutting down system components...")
            
            # Stop web interface
            if 'web_interface' in self.components:
                self.components['web_interface'].stop_server()
            
            # Stop KoboldCPP server
            if 'kobold_integration' in self.components:
                await self.components['kobold_integration'].stop_kobold_server()
            
            # Stop tensor memory system
            if 'tensor_memory' in self.components:
                self.components['tensor_memory'].stop_memory_system()
            
            logger.info("✅ System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

async def main():
    """Main function."""
    try:
        # Create launcher
        launcher = KoboldCPPCompleteSystemLauncher()
        
        # Run system
        success = await launcher.run_system()
        
        if success:
            logger.info("✅ System completed successfully")
        else:
            logger.error("❌ System failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if running in the correct directory
    if not Path("core").exists():
        print("❌ Error: Please run this script from the AOI_Base_Files_Schwabot directory")
        sys.exit(1)
    
    # Run the system
    asyncio.run(main()) 