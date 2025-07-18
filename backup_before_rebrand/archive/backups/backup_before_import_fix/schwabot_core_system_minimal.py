#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Core System - Minimal Orchestrator
===========================================

Minimal version that only imports essential components that are known to work.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from utils.logging_setup import setup_logging
from utils.secure_config_manager import SecureConfigManager

# Core imports - only essential components
from .btc_usdc_trading_engine import BTCTradingEngine
from .enhanced_mathematical_core import EnhancedMathematicalCore
from .math_config_manager import MathConfigManager
from .mathematical_framework_integrator import MathematicalFrameworkIntegrator
from .profit_optimization_engine import ProfitOptimizationEngine
from .real_multi_exchange_trader import RealMultiExchangeTrader
from .risk_manager import RiskManager
from .secure_exchange_manager import SecureExchangeManager
from .strategy.strategy_executor import StrategyExecutor

# Strategy imports
from .strategy.strategy_loader import StrategyLoader
from .tcell_survival_engine import TCellSurvivalEngine

# Utility imports
from .type_defs import OrderSide, OrderType, TradingMode, TradingPair
from .unified_btc_trading_pipeline import UnifiedBTCTradingPipeline
from .unified_pipeline_manager import UnifiedPipelineManager

logger = logging.getLogger(__name__)


class SubsystemWrapper:
    """Wrapper for subsystems to normalize their interfaces."""
    
    def __init__(self, name: str, instance: Any, config: Dict[str, Any] = None):
        self.name = name
        self.instance = instance
        self.config = config or {}
        self.is_initialized = False
        self.is_running = False
        self.last_entropy_check = 0
        self.entropy_threshold = 0.7
        
    async def initialize(self) -> bool:
        """Initialize the subsystem."""
        try:
            if hasattr(self.instance, 'initialize'):
                if asyncio.iscoroutinefunction(self.instance.initialize):
                    await self.instance.initialize()
                else:
                    self.instance.initialize()
                self.is_initialized = True
                logger.info(f"✅ {self.name} initialized")
                return True
            elif hasattr(self.instance, 'initialized') and self.instance.initialized:
                self.is_initialized = True
                logger.info(f"✅ {self.name} already initialized")
                return True
            elif hasattr(self.instance, 'active') and self.instance.active:
                self.is_initialized = True
                logger.info(f"✅ {self.name} already active")
                return True
            else:
                # Assume it's initialized if no init method
                self.is_initialized = True
                logger.info(f"✅ {self.name} initialized (no init method)")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.name}: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the subsystem."""
        try:
            if hasattr(self.instance, 'start'):
                if asyncio.iscoroutinefunction(self.instance.start):
                    await self.instance.start()
                else:
                    self.instance.start()
                self.is_running = True
                logger.info(f"✅ {self.name} started")
                return True
            else:
                self.is_running = True
                logger.info(f"✅ {self.name} started (no start method)")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to start {self.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the subsystem."""
        try:
            if hasattr(self.instance, 'stop'):
                if asyncio.iscoroutinefunction(self.instance.stop):
                    await self.instance.stop()
                else:
                    self.instance.stop()
            elif hasattr(self.instance, 'deactivate'):
                self.instance.deactivate()
            
            self.is_running = False
            logger.info(f"✅ {self.name} stopped")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to stop {self.name}: {e}")
            return False
    
    async def reload(self) -> bool:
        """Reload the subsystem."""
        try:
            if hasattr(self.instance, 'reload'):
                if asyncio.iscoroutinefunction(self.instance.reload):
                    await self.instance.reload()
                else:
                    self.instance.reload()
                logger.info(f"✅ {self.name} reloaded")
                return True
            else:
                # Stop and reinitialize
                await self.stop()
                await self.initialize()
                await self.start()
                logger.info(f"✅ {self.name} reloaded (stop/init/start)")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to reload {self.name}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get subsystem status."""
        status = {
            'name': self.name,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'config': self.config
        }
        
        # Add instance-specific status if available
        if hasattr(self.instance, 'get_status'):
            try:
                instance_status = self.instance.get_status()
                status.update(instance_status)
            except Exception as e:
                logger.warning(f"Failed to get status for {self.name}: {e}")
        
        return status
    
    def check_entropy_change(self) -> bool:
        """Check if entropy has changed significantly."""
        try:
            if hasattr(self.instance, 'get_entropy'):
                current_entropy = self.instance.get_entropy()
                if abs(current_entropy - self.last_entropy_check) > self.entropy_threshold:
                    self.last_entropy_check = current_entropy
                    return True
        except Exception as e:
            logger.debug(f"Entropy check failed for {self.name}: {e}")
        
        return False


class SchwabotCoreSystem:
    """
    Minimal orchestrator for the Schwabot trading system.
    
    This class wraps and injects essential subsystems with normalized interfaces,
    handles hot reloading, and provides both CLI and API access.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Schwabot core system."""
        self.config_path = config_path or "config/schwabot_config.yaml"
        self.config = self._load_config()
        
        # System state
        self.is_running = False
        self.is_initialized = False
        self.start_time = None
        self.subsystems: Dict[str, SubsystemWrapper] = {}
        
        # Configuration
        self.secure_config = SecureConfigManager()
        
        # Setup logging
        setup_logging(
            level=self.config.get("logging", {}).get("level", "INFO"),
            log_file=self.config.get("logging", {}).get("file", "logs/schwabot.log")
        )
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        logger.info("Schwabot Core System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "system": {
                "name": "Schwabot Trading System",
                "version": "2.1.0",
                "environment": "development"
            },
            "trading": {
                "mode": "sandbox",
                "default_symbol": "BTC/USDT",
                "max_positions": 5,
                "base_capital": 10000.0
            },
            "risk_management": {
                "max_position_size_pct": 10.0,
                "max_total_exposure_pct": 30.0,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 5.0,
                "max_daily_loss_usd": 1000.0
            },
            "exchanges": {
                "primary": ["binance", "coinbase"],
                "paper_trading": True
            },
            "mathematical_engine": {
                "tensor_depth": 4,
                "hash_memory_depth": 100,
                "quantum_dimension": 16,
                "entropy_threshold": 0.7
            },
            "logging": {
                "level": "INFO",
                "file": "logs/schwabot.log"
            }
        }
    
    def _initialize_subsystems(self):
        """Initialize all subsystem wrappers."""
        logger.info("Initializing subsystem wrappers...")
        
        # Define essential subsystems with their configurations
        subsystem_definitions = [
            # Mathematical components
            ("MathConfigManager", MathConfigManager, {}),
            ("EnhancedMathematicalCore", EnhancedMathematicalCore, {}),
            ("MathematicalFrameworkIntegrator", MathematicalFrameworkIntegrator, {}),
            ("TCellSurvivalEngine", TCellSurvivalEngine, {}),
            
            # Trading components
            ("BTCTradingEngine", BTCTradingEngine, {
                "api_key": os.getenv("BINANCE_API_KEY", "demo"),
                "api_secret": os.getenv("BINANCE_API_SECRET", "demo"),
                "testnet": self.config.get("trading", {}).get("mode") == "sandbox",
                "symbol": self.config.get("trading", {}).get("default_symbol", "BTC/USDT")
            }),
            ("RiskManager", RiskManager, {}),
            ("SecureExchangeManager", SecureExchangeManager, {}),
            ("UnifiedPipelineManager", UnifiedPipelineManager, {}),
            ("UnifiedBTCTradingPipeline", UnifiedBTCTradingPipeline, {}),
            ("ProfitOptimizationEngine", ProfitOptimizationEngine, {}),
            ("RealMultiExchangeTrader", RealMultiExchangeTrader, {}),
            
            # Strategy components
            ("StrategyLoader", StrategyLoader, {}),
            ("StrategyExecutor", StrategyExecutor, {}),
        ]
        
        # Create subsystem wrappers
        for name, cls, config in subsystem_definitions:
            try:
                instance = cls(config)
                wrapper = SubsystemWrapper(name, instance, config)
                self.subsystems[name] = wrapper
                logger.debug(f"Created subsystem wrapper: {name}")
            except Exception as e:
                logger.warning(f"Failed to create subsystem {name}: {e}")
        
        logger.info(f"Initialized {len(self.subsystems)} subsystem wrappers")
    
    async def initialize(self) -> bool:
        """Initialize all subsystems."""
        try:
            logger.info("Initializing Schwabot Core System...")
            
            # Initialize all subsystems
            success_count = 0
            total_count = len(self.subsystems)
            
            for name, wrapper in self.subsystems.items():
                try:
                    if await wrapper.initialize():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to initialize {name}: {e}")
            
            logger.info(f"Initialized {success_count}/{total_count} subsystems")
            
            if success_count > 0:
                self.is_initialized = True
                logger.info("Schwabot Core System initialized successfully")
                return True
            else:
                logger.error("No subsystems initialized successfully")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def start(self) -> bool:
        """Start all subsystems."""
        if not self.is_initialized:
            logger.error("System not initialized. Call initialize() first.")
            return False
        
        try:
            logger.info("Starting Schwabot Core System...")
            
            # Start all subsystems
            success_count = 0
            total_count = len(self.subsystems)
            
            for name, wrapper in self.subsystems.items():
                try:
                    if await wrapper.start():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to start {name}: {e}")
            
            logger.info(f"Started {success_count}/{total_count} subsystems")
            
            if success_count > 0:
                self.is_running = True
                self.start_time = datetime.now()
                logger.info("Schwabot Core System started successfully")
                return True
            else:
                logger.error("No subsystems started successfully")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    async def stop(self):
        """Stop all subsystems."""
        if not self.is_running:
            return
        
        logger.info("Stopping Schwabot Core System...")
        
        try:
            # Stop all subsystems
            for name, wrapper in self.subsystems.items():
                try:
                    await wrapper.stop()
                except Exception as e:
                    logger.error(f"Failed to stop {name}: {e}")
            
            self.is_running = False
            logger.info("Schwabot Core System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    async def reload_subsystem(self, subsystem_name: str) -> bool:
        """Reload a specific subsystem."""
        if subsystem_name not in self.subsystems:
            logger.error(f"Subsystem {subsystem_name} not found")
            return False
        
        try:
            wrapper = self.subsystems[subsystem_name]
            success = await wrapper.reload()
            if success:
                logger.info(f"✅ {subsystem_name} reloaded successfully")
            else:
                logger.error(f"❌ Failed to reload {subsystem_name}")
            return success
        except Exception as e:
            logger.error(f"Error reloading {subsystem_name}: {e}")
            return False
    
    async def reload_all_subsystems(self) -> bool:
        """Reload all subsystems."""
        logger.info("Reloading all subsystems...")
        
        try:
            success_count = 0
            total_count = len(self.subsystems)
            
            for name, wrapper in self.subsystems.items():
                try:
                    if await wrapper.reload():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to reload {name}: {e}")
            
            logger.info(f"Reloaded {success_count}/{total_count} subsystems")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error reloading subsystems: {e}")
            return False
    
    async def check_entropy_changes(self) -> List[str]:
        """Check for entropy changes in subsystems."""
        changed_subsystems = []
        
        for name, wrapper in self.subsystems.items():
            if wrapper.check_entropy_change():
                changed_subsystems.append(name)
        
        if changed_subsystems:
            logger.info(f"Entropy changes detected in: {changed_subsystems}")
        
        return changed_subsystems
    
    async def run_trading_loop(self):
        """Main trading loop with entropy monitoring."""
        if not self.is_running:
            logger.error("System not running. Call start() first.")
            return
        
        logger.info("Starting main trading loop...")
        
        try:
            while self.is_running:
                # Check for entropy changes
                entropy_changes = await self.check_entropy_changes()
                
                # Reload subsystems with entropy changes
                for subsystem_name in entropy_changes:
                    await self.reload_subsystem(subsystem_name)
                
                # Simulate trading operations
                await self._execute_trading_cycle()
                
                # Sleep for next iteration
                await asyncio.sleep(1)  # 1 second interval
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await self.stop()
    
    async def _execute_trading_cycle(self):
        """Execute one trading cycle."""
        try:
            # Get market data
            market_data = await self._get_market_data()
            
            # Analyze market conditions
            analysis = await self._analyze_market(market_data)
            
            # Generate trading signals
            signals = await self._generate_signals(analysis)
            
            # Execute trades
            if signals:
                await self._execute_trades(signals)
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get market data from subsystems."""
        market_data = {"timestamp": datetime.now(), "price": 50000.0}
        return market_data
    
    async def _analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions using mathematical subsystems."""
        analysis = {}
        
        # Use mathematical analysis subsystems
        math_subsystems = [
            "EnhancedMathematicalCore",
            "TCellSurvivalEngine"
        ]
        
        for name in math_subsystems:
            if name in self.subsystems:
                try:
                    wrapper = self.subsystems[name]
                    if hasattr(wrapper.instance, 'analyze'):
                        result = await wrapper.instance.analyze(market_data)
                        analysis[name] = result
                except Exception as e:
                    logger.debug(f"Analysis error in {name}: {e}")
        
        return analysis
    
    async def _generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals using strategy subsystems."""
        signals = []
        
        # Use strategy subsystems
        strategy_subsystems = [
            "StrategyExecutor",
            "MathematicalFrameworkIntegrator",
            "ProfitOptimizationEngine"
        ]
        
        for name in strategy_subsystems:
            if name in self.subsystems:
                try:
                    wrapper = self.subsystems[name]
                    if hasattr(wrapper.instance, 'generate_signals'):
                        result = await wrapper.instance.generate_signals(analysis)
                        signals.extend(result)
                except Exception as e:
                    logger.debug(f"Signal generation error in {name}: {e}")
        
        return signals
    
    async def _execute_trades(self, signals: List[Dict[str, Any]]):
        """Execute trades using trading subsystems."""
        for signal in signals:
            try:
                # Risk check
                if "RiskManager" in self.subsystems:
                    wrapper = self.subsystems["RiskManager"]
                    if hasattr(wrapper.instance, 'validate_signal'):
                        if not await wrapper.instance.validate_signal(signal):
                            logger.warning(f"Signal rejected by risk manager: {signal}")
                            continue
                
                # Execute trade
                if "BTCTradingEngine" in self.subsystems:
                    wrapper = self.subsystems["BTCTradingEngine"]
                    if hasattr(wrapper.instance, 'execute_signal'):
                        result = await wrapper.instance.execute_signal(signal)
                        logger.info(f"Trade executed: {result}")
                
            except Exception as e:
                logger.error(f"Error executing trade: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else None,
            "subsystems": {}
        }
        
        # Add status for each subsystem
        for name, wrapper in self.subsystems.items():
            status["subsystems"][name] = wrapper.get_status()
        
        return status
    
    def get_subsystem(self, name: str) -> Optional[Any]:
        """Get a subsystem instance by name."""
        if name in self.subsystems:
            return self.subsystems[name].instance
        return None
    
    def list_subsystems(self) -> List[str]:
        """List all subsystem names."""
        return list(self.subsystems.keys())
    
    async def call_subsystem_method(self, subsystem_name: str, method_name: str, *args, **kwargs) -> Any:
        """Call a method on a specific subsystem."""
        if subsystem_name not in self.subsystems:
            raise ValueError(f"Subsystem {subsystem_name} not found")
        
        wrapper = self.subsystems[subsystem_name]
        instance = wrapper.instance
        
        if not hasattr(instance, method_name):
            raise ValueError(f"Method {method_name} not found on {subsystem_name}")
        
        method = getattr(instance, method_name)
        
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            return method(*args, **kwargs)
    
    # CLI and API methods
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                         quantity: Decimal, price: Optional[Decimal] = None) -> Dict[str, Any]:
        """Place a trading order via CLI/API."""
        if not self.is_running:
            raise RuntimeError("System not running")
        
        if "BTCTradingEngine" in self.subsystems:
            return await self.call_subsystem_method(
                "BTCTradingEngine", "place_order", symbol, side, order_type, quantity, price
            )
        
        raise RuntimeError("Trading engine not available")
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status via CLI/API."""
        if "BTCTradingEngine" in self.subsystems:
            return await self.call_subsystem_method("BTCTradingEngine", "get_order_status", order_id)
        
        raise RuntimeError("Trading engine not available")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order via CLI/API."""
        if "BTCTradingEngine" in self.subsystems:
            return await self.call_subsystem_method("BTCTradingEngine", "cancel_order", order_id)
        
        raise RuntimeError("Trading engine not available")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary via CLI/API."""
        return {}


# Global system instance
_system_instance: Optional[SchwabotCoreSystem] = None


def get_system_instance() -> Optional[SchwabotCoreSystem]:
    """Get the global system instance."""
    return _system_instance


def create_system_instance(config_path: Optional[str] = None) -> SchwabotCoreSystem:
    """Create and return a new system instance."""
    global _system_instance
    _system_instance = SchwabotCoreSystem(config_path)
    return _system_instance


async def run_system(config_path: Optional[str] = None):
    """Run the Schwabot system."""
    system = create_system_instance(config_path)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        if not await system.initialize():
            logger.error("Failed to initialize system")
            return
        
        # Start system
        if not await system.start():
            logger.error("Failed to start system")
            return
        
        # Run trading loop
        await system.run_trading_loop()
        
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.stop()


if __name__ == "__main__":
    # Run the system
    asyncio.run(run_system()) 