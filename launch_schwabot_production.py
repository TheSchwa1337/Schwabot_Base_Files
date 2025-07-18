#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SCHWABOT PRODUCTION LAUNCHER - COMPLETE TRADING SYSTEM
========================================================

Main launcher for the complete Schwabot trading system.
This starts ALL components:

1. Real Trading Engine with API connections
2. Cascade Memory Architecture
3. Web Interface with real-time dashboard
4. Backtesting System
5. Risk Management
6. GUFF AI Integration

This is the COMPLETE production system that performs REAL trades
using REAL market data and REAL mathematical models.

Usage:
    python launch_schwabot_production.py --config config.json
    python launch_schwabot_production.py --sandbox  # For testing
    python launch_schwabot_production.py --live     # For live trading
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SchwabotProductionLauncher:
    """
    Production launcher for the complete Schwabot trading system.
    
    This orchestrates all components and provides a unified interface
    for starting, stopping, and monitoring the entire system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.components = {}
        
        # Import components
        self._import_components()
        
        logger.info("🚀 Schwabot Production Launcher initialized")
    
    def _import_components(self):
        """Import all Schwabot components."""
        try:
            # Import core components
            from core.real_trading_engine import RealTradingEngine
            from core.cascade_memory_architecture import CascadeMemoryArchitecture
            from core.lantern_core_risk_profiles import LanternCoreRiskProfiles
            from core.trade_gating_system import TradeGatingSystem
            from core.real_backtesting_system import RealBacktestingSystem
            from mathlib.mathlib_v4 import MathLibV4
            
            # Import web interface
            from web.schwabot_trading_interface import SchwabotWebInterface
            
            # Store component classes
            self.components = {
                'trading_engine': RealTradingEngine,
                'cascade_memory': CascadeMemoryArchitecture,
                'risk_profiles': LanternCoreRiskProfiles,
                'trade_gating': TradeGatingSystem,
                'backtesting': RealBacktestingSystem,
                'math_lib': MathLibV4,
                'web_interface': SchwabotWebInterface
            }
            
            logger.info("🚀 All Schwabot components imported successfully")
            
        except ImportError as e:
            logger.error(f"Error importing components: {e}")
            raise
    
    async def start_system(self):
        """Start the complete Schwabot trading system."""
        try:
            logger.info("🚀 Starting Schwabot Production System...")
            
            self.running = True
            
            # Initialize trading engine
            await self._start_trading_engine()
            
            # Initialize cascade memory
            await self._start_cascade_memory()
            
            # Initialize risk management
            await self._start_risk_management()
            
            # Initialize web interface
            await self._start_web_interface()
            
            # Start real-time data feeds
            await self._start_data_feeds()
            
            # Start monitoring
            await self._start_monitoring()
            
            logger.info("🚀 Schwabot Production System started successfully!")
            logger.info("🌐 Web Interface: http://localhost:5000")
            logger.info("📊 Dashboard: http://localhost:5000/dashboard")
            logger.info("⚡ Trading: http://localhost:5000/trading")
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            await self.stop_system()
            raise
    
    async def _start_trading_engine(self):
        """Start the real trading engine."""
        try:
            logger.info("🚀 Initializing Real Trading Engine...")
            
            # Create trading engine configuration
            trading_config = {
                'api_keys': self.config.get('api_keys', {}),
                'secret_keys': self.config.get('secret_keys', {}),
                'passphrases': self.config.get('passphrases', {}),
                'sandbox_mode': self.config.get('sandbox_mode', True),
                'initial_capital': self.config.get('initial_capital', 10000.0),
                'cascade_config': self.config.get('cascade_config', {})
            }
            
            # Initialize trading engine
            self.trading_engine = self.components['trading_engine'](trading_config)
            
            # Initialize exchange connections
            success = await self.trading_engine.initialize_exchanges()
            
            if success:
                logger.info("🚀 Trading Engine started successfully")
            else:
                logger.warning("🚀 Trading Engine started in offline mode (no API credentials)")
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            raise
    
    async def _start_cascade_memory(self):
        """Start the cascade memory architecture."""
        try:
            logger.info("🌊 Initializing Cascade Memory Architecture...")
            
            # Create cascade memory configuration
            cascade_config = self.config.get('cascade_config', {
                'echo_decay_factor': 0.1,
                'cascade_threshold': 0.7,
                'memory_decay': 0.95,
                'echo_resonance_threshold': 0.8
            })
            
            # Initialize cascade memory
            self.cascade_memory = self.components['cascade_memory'](cascade_config)
            
            logger.info("🌊 Cascade Memory Architecture started successfully")
            
        except Exception as e:
            logger.error(f"Error starting cascade memory: {e}")
            raise
    
    async def _start_risk_management(self):
        """Start the risk management system."""
        try:
            logger.info("🛡️ Initializing Risk Management System...")
            
            # Initialize risk profiles
            self.risk_profiles = self.components['risk_profiles']()
            
            # Initialize trade gating
            self.trade_gating = self.components['trade_gating']()
            
            # Initialize math library
            self.math_lib = self.components['math_lib']()
            
            logger.info("🛡️ Risk Management System started successfully")
            
        except Exception as e:
            logger.error(f"Error starting risk management: {e}")
            raise
    
    async def _start_web_interface(self):
        """Start the web interface."""
        try:
            logger.info("🌐 Initializing Web Interface...")
            
            # Create web interface configuration
            web_config = {
                'api_keys': self.config.get('api_keys', {}),
                'secret_keys': self.config.get('secret_keys', {}),
                'passphrases': self.config.get('passphrases', {}),
                'sandbox_mode': self.config.get('sandbox_mode', True),
                'initial_capital': self.config.get('initial_capital', 10000.0),
                'cascade_config': self.config.get('cascade_config', {})
            }
            
            # Initialize web interface
            self.web_interface = self.components['web_interface'](web_config)
            
            # Initialize trading engine in web interface
            await self.web_interface.initialize_trading_engine()
            
            # Start data collection thread
            self.web_interface.start_data_thread()
            
            logger.info("🌐 Web Interface started successfully")
            
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            raise
    
    async def _start_data_feeds(self):
        """Start real-time data feeds."""
        try:
            logger.info("📡 Starting Real-Time Data Feeds...")
            
            if hasattr(self.trading_engine, 'start_real_time_data_feeds'):
                await self.trading_engine.start_real_time_data_feeds()
                logger.info("📡 Real-time data feeds started")
            else:
                logger.warning("📡 Real-time data feeds not available")
            
        except Exception as e:
            logger.error(f"Error starting data feeds: {e}")
    
    async def _start_monitoring(self):
        """Start system monitoring."""
        try:
            logger.info("📊 Starting System Monitoring...")
            
            # Start monitoring task
            asyncio.create_task(self._monitor_system())
            
            logger.info("📊 System monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
    
    async def _monitor_system(self):
        """Monitor system health and performance."""
        try:
            while self.running:
                try:
                    # Check trading engine status
                    if hasattr(self.trading_engine, 'get_portfolio_status'):
                        portfolio = await self.trading_engine.get_portfolio_status()
                        if 'error' not in portfolio:
                            logger.debug(f"📊 Portfolio: ${portfolio.get('total_value', 0):.2f}")
                    
                    # Check cascade memory status
                    if hasattr(self.cascade_memory, 'get_system_status'):
                        cascade_status = self.cascade_memory.get_system_status()
                        if 'error' not in cascade_status:
                            logger.debug(f"🌊 Cascades: {cascade_status.get('total_cascades', 0)}")
                    
                    # Check web interface status
                    if hasattr(self.web_interface, 'running'):
                        if self.web_interface.running:
                            logger.debug("🌐 Web interface: Running")
                        else:
                            logger.warning("🌐 Web interface: Stopped")
                    
                    # Wait before next check
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def stop_system(self):
        """Stop the complete Schwabot trading system."""
        try:
            logger.info("🚀 Stopping Schwabot Production System...")
            
            self.running = False
            
            # Stop web interface
            if hasattr(self, 'web_interface') and self.web_interface:
                self.web_interface.stop_data_thread()
                logger.info("🌐 Web interface stopped")
            
            # Stop data feeds
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self.trading_engine, 'stop_real_time_data_feeds'):
                    await self.trading_engine.stop_real_time_data_feeds()
                    logger.info("📡 Data feeds stopped")
            
            logger.info("🚀 Schwabot Production System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'running': self.running,
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }
            
            # Check trading engine
            if hasattr(self, 'trading_engine') and self.trading_engine:
                try:
                    portfolio = await self.trading_engine.get_portfolio_status()
                    status['components']['trading_engine'] = {
                        'status': 'running' if 'error' not in portfolio else 'error',
                        'portfolio_value': portfolio.get('total_value', 0),
                        'total_trades': portfolio.get('total_trades', 0)
                    }
                except Exception as e:
                    status['components']['trading_engine'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                status['components']['trading_engine'] = {'status': 'not_initialized'}
            
            # Check cascade memory
            if hasattr(self, 'cascade_memory') and self.cascade_memory:
                try:
                    cascade_status = self.cascade_memory.get_system_status()
                    status['components']['cascade_memory'] = {
                        'status': 'running' if 'error' not in cascade_status else 'error',
                        'total_cascades': cascade_status.get('total_cascades', 0),
                        'success_rate': cascade_status.get('success_rate', 0)
                    }
                except Exception as e:
                    status['components']['cascade_memory'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                status['components']['cascade_memory'] = {'status': 'not_initialized'}
            
            # Check web interface
            if hasattr(self, 'web_interface') and self.web_interface:
                status['components']['web_interface'] = {
                    'status': 'running' if self.web_interface.running else 'stopped',
                    'data_collection': self.web_interface.running
                }
            else:
                status['components']['web_interface'] = {'status': 'not_initialized'}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"📄 Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'sandbox_mode': True,
        'initial_capital': 10000.0,
        'api_keys': {
            'coinbase': 'your_coinbase_api_key',
            'binance': 'your_binance_api_key',
            'kraken': 'your_kraken_api_key'
        },
        'secret_keys': {
            'coinbase': 'your_coinbase_secret',
            'binance': 'your_binance_secret',
            'kraken': 'your_kraken_secret'
        },
        'passphrases': {
            'coinbase': 'your_coinbase_passphrase'
        },
        'cascade_config': {
            'echo_decay_factor': 0.1,
            'cascade_threshold': 0.7,
            'memory_decay': 0.95,
            'echo_resonance_threshold': 0.8
        },
        'risk_config': {
            'default_profile': 'blue',
            'max_risk_per_trade': 0.02,
            'max_portfolio_risk': 0.1
        },
        'web_config': {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False
        }
    }

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"📄 Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Launch Schwabot Production System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--sandbox', action='store_true', help='Run in sandbox mode')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    parser.add_argument('--create-config', type=str, help='Create default config file')
    parser.add_argument('--status', action='store_true', help='Show system status')
    
    args = parser.parse_args()
    
    try:
        # Handle create-config
        if args.create_config:
            config = create_default_config()
            save_config(config, args.create_config)
            print(f"📄 Default configuration created: {args.create_config}")
            return
        
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            # Use default config
            config = create_default_config()
        
        # Override sandbox mode if specified
        if args.sandbox:
            config['sandbox_mode'] = True
        elif args.live:
            config['sandbox_mode'] = False
        
        # Create launcher
        launcher = SchwabotProductionLauncher(config)
        
        # Handle status check
        if args.status:
            status = await launcher.get_system_status()
            print(json.dumps(status, indent=2))
            return
        
        # Start system
        await launcher.start_system()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🚀 Received shutdown signal")
        finally:
            await launcher.stop_system()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"🚀 Received signal {signum}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run main
    asyncio.run(main()) 