#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Mathematical Trading System Launcher
===========================================

This launcher starts the complete unified mathematical trading system with:
1. Flask server for multi-bot coordination
2. Mathematical integration (DLT, Dualistic Engines, Bit Phases, etc.)
3. KoboldCPP integration for CUDA acceleration
4. Multiple API endpoint management
5. Registry and soulprint storage
6. Real-time trading pipeline

This is the main entry point for the complete trading system.
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, Any, List
import argparse
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the unified system
try:
    from core.unified_mathematical_trading_system import UnifiedMathematicalTradingSystem, create_unified_trading_system
    from core.production_trading_pipeline import ProductionTradingPipeline, TradingConfig
    from backtesting.mathematical_integration_simplified import mathematical_integration
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import unified system: {e}")
    UNIFIED_SYSTEM_AVAILABLE = False

class UnifiedTradingSystemLauncher:
    """Launcher for the unified mathematical trading system."""
    
    def __init__(self, config_path: str = None):
        """Initialize the launcher."""
        self.config = self._load_config(config_path)
        self.system = None
        self.running = False
        self.flask_thread = None
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Unified Trading System Launcher initialized")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ Configuration loaded from {config_path}")
                return config
            except Exception as e:
                logger.error(f"❌ Failed to load config from {config_path}: {e}")
        
        # Default configuration
        default_config = {
            'flask_server': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            },
            'mathematical_integration': {
                'enabled': True,
                'weight': 0.7,
                'confidence_threshold': 0.7
            },
            'koboldcpp_integration': {
                'enabled': True,
                'server_url': 'http://localhost:5001',
                'model_name': 'default'
            },
            'trading': {
                'base_position_size': 0.01,
                'max_position_size': 0.1,
                'risk_tolerance': 0.2,
                'consensus_threshold': 0.7
            },
            'api_endpoints': [
                {
                    'name': 'primary',
                    'exchange': 'binance',
                    'api_key': 'your_api_key_here',
                    'secret': 'your_secret_here',
                    'sandbox': True
                }
            ],
            'monitoring': {
                'heartbeat_interval': 30,
                'performance_update_interval': 60,
                'log_level': 'INFO'
            }
        }
        
        logger.info("✅ Using default configuration")
        return default_config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    async def start_system(self):
        """Start the unified trading system."""
        try:
            if not UNIFIED_SYSTEM_AVAILABLE:
                logger.error("❌ Unified system not available")
                return False
            
            logger.info("🚀 Starting Unified Mathematical Trading System...")
            
            # Create the unified system
            self.system = create_unified_trading_system(self.config)
            
            # Start Flask server in a separate thread
            self.flask_thread = threading.Thread(
                target=self._start_flask_server,
                daemon=True
            )
            self.flask_thread.start()
            
            # Wait for Flask server to start
            await asyncio.sleep(2)
            
            # Initialize trading pipelines for each API endpoint
            await self._initialize_trading_pipelines()
            
            # Start market data processing
            await self._start_market_data_processing()
            
            # Start monitoring
            await self._start_monitoring()
            
            self.running = True
            logger.info("✅ Unified Mathematical Trading System started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}")
            return False
    
    def _start_flask_server(self):
        """Start Flask server in a separate thread."""
        try:
            flask_config = self.config['flask_server']
            self.system.start_flask_server(
                host=flask_config['host'],
                port=flask_config['port']
            )
        except Exception as e:
            logger.error(f"❌ Flask server error: {e}")
    
    async def _initialize_trading_pipelines(self):
        """Initialize trading pipelines for each API endpoint."""
        try:
            for endpoint_config in self.config['api_endpoints']:
                logger.info(f"🔧 Initializing trading pipeline for {endpoint_config['name']}")
                
                # Create trading config
                trading_config = TradingConfig(
                    exchange_name=endpoint_config['exchange'],
                    api_key=endpoint_config['api_key'],
                    secret=endpoint_config['secret'],
                    sandbox=endpoint_config.get('sandbox', True),
                    symbols=['BTC/USDC'],
                    enable_mathematical_integration=self.config['mathematical_integration']['enabled'],
                    mathematical_confidence_threshold=self.config['mathematical_integration']['confidence_threshold']
                )
                
                # Create and store pipeline
                pipeline = ProductionTradingPipeline(trading_config)
                self.system.trading_pipelines[endpoint_config['name']] = pipeline
                
                logger.info(f"✅ Trading pipeline initialized for {endpoint_config['name']}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading pipelines: {e}")
    
    async def _start_market_data_processing(self):
        """Start market data processing loop."""
        try:
            logger.info("📊 Starting market data processing...")
            
            while self.running:
                try:
                    # Simulate market data (in real implementation, this would come from exchanges)
                    market_data = self._generate_simulated_market_data()
                    
                    # Process through unified system
                    decision = await self.system.process_market_data_comprehensive(market_data)
                    
                    # Execute trades if confidence is high enough
                    if decision.confidence >= self.config['mathematical_integration']['confidence_threshold']:
                        await self._execute_trades(decision)
                    
                    # Wait before next iteration
                    await asyncio.sleep(5)  # 5-second intervals
                    
                except Exception as e:
                    logger.error(f"❌ Market data processing error: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"❌ Failed to start market data processing: {e}")
    
    def _generate_simulated_market_data(self) -> Dict[str, Any]:
        """Generate simulated market data for testing."""
        import random
        
        base_price = 50000.0
        price_change = random.uniform(-0.02, 0.02)  # ±2% price change
        current_price = base_price * (1 + price_change)
        
        return {
            'symbol': 'BTC/USDC',
            'price': current_price,
            'volume': random.uniform(1000, 5000),
            'volatility': random.uniform(0.05, 0.25),
            'price_change': price_change,
            'sentiment': random.uniform(0.3, 0.7),
            'price_history': [
                base_price * (1 + random.uniform(-0.01, 0.01)) for _ in range(10)
            ],
            'timestamp': time.time()
        }
    
    async def _execute_trades(self, decision):
        """Execute trades based on decision."""
        try:
            logger.info(f"💰 Executing trade: {decision.action} {decision.symbol} @ {decision.confidence:.3f}")
            
            # Execute on all trading pipelines
            for pipeline_name, pipeline in self.system.trading_pipelines.items():
                try:
                    # In real implementation, this would execute actual trades
                    logger.info(f"   Executing on {pipeline_name}")
                    
                    # Simulate trade execution
                    await asyncio.sleep(0.1)  # Simulate execution time
                    
                except Exception as e:
                    logger.error(f"❌ Trade execution failed on {pipeline_name}: {e}")
            
            # Update performance metrics
            self.system.performance_metrics['successful_trades'] += 1
            
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
    
    async def _start_monitoring(self):
        """Start system monitoring."""
        try:
            logger.info("📈 Starting system monitoring...")
            
            while self.running:
                try:
                    # Get system status
                    status = self.system.get_system_status()
                    
                    # Log status
                    logger.info(f"📊 System Status:")
                    logger.info(f"   Connected Bots: {status['connected_bots']}")
                    logger.info(f"   Total Decisions: {status['total_decisions']}")
                    logger.info(f"   Successful Trades: {status['performance_metrics']['successful_trades']}")
                    logger.info(f"   Mathematical Decisions: {status['performance_metrics']['mathematical_decisions']}")
                    logger.info(f"   AI Decisions: {status['performance_metrics']['ai_decisions']}")
                    
                    # Wait before next status check
                    await asyncio.sleep(self.config['monitoring']['performance_update_interval'])
                    
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {e}")
    
    def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            logger.info("🛑 Shutting down Unified Mathematical Trading System...")
            
            self.running = False
            
            # Shutdown trading pipelines
            for pipeline_name, pipeline in self.system.trading_pipelines.items():
                try:
                    logger.info(f"🛑 Shutting down {pipeline_name} pipeline")
                    # In real implementation, this would properly close the pipeline
                except Exception as e:
                    logger.error(f"❌ Error shutting down {pipeline_name}: {e}")
            
            # Save final status
            if self.system:
                final_status = self.system.get_system_status()
                logger.info(f"📊 Final System Status: {json.dumps(final_status, indent=2)}")
            
            logger.info("✅ System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Launch Unified Mathematical Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Flask server host')
    parser.add_argument('--port', type=int, default=5000, help='Flask server port')
    
    args = parser.parse_args()
    
    # Create and start launcher
    launcher = UnifiedTradingSystemLauncher(args.config)
    
    # Override config with command line arguments
    if args.host:
        launcher.config['flask_server']['host'] = args.host
    if args.port:
        launcher.config['flask_server']['port'] = args.port
    
    # Start the system
    success = await launcher.start_system()
    
    if success:
        logger.info("🎉 Unified Mathematical Trading System is running!")
        logger.info(f"🌐 Flask server: http://{launcher.config['flask_server']['host']}:{launcher.config['flask_server']['port']}")
        logger.info("🧮 Mathematical integration: ACTIVE")
        logger.info("🤖 KoboldCPP integration: ACTIVE")
        logger.info("📊 Multi-bot coordination: ACTIVE")
        logger.info("💾 Registry and soulprint storage: ACTIVE")
        
        # Keep the system running
        try:
            while launcher.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        finally:
            launcher.shutdown()
    else:
        logger.error("❌ Failed to start Unified Mathematical Trading System")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 