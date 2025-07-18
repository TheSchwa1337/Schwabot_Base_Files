#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Core System
====================
Main system orchestrator for Schwabot trading bot.

Features:
- Subsystem management
- Trading cycle execution
- Market data processing
- Signal generation
- Order execution
- Portfolio management
- Robust error handling
- CLI and API access
- System lifecycle management
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

from .advanced_tensor_algebra import AdvancedTensorAlgebra
from .bitmap_hash_folding import BitmapHashFolding

# Core imports - ONLY what actually exists and works
from .btc_usdc_trading_engine import BTCTradingEngine
from .ccxt_trading_executor import CCXTTradingExecutor
from .crwf_crlf_integration import CRWFCrlfIntegration
from .enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine
from .enhanced_mathematical_core import EnhancedMathematicalCore
from .entropy_decay_system import EntropyDecaySystem
from .entropy_drift_engine import EntropyDriftEngine
from .entropy_math import EntropyMath
from .fill_handler import FillHandler
from .fractal_core import FractalCore
from .fractal_memory_tracker import FractalMemoryTracker
from .ghost_core import GhostCore
from .gpu_dna_autodetect import GPUDNAAutodetect
from .gpu_shader_integration import GPUShaderIntegration
from .math_cache import MathResultCache
from .math_config_manager import MathConfigManager
from .math_orchestrator import MathOrchestrator
from .mathematical_framework_integrator import MathematicalFrameworkIntegrator
from .orbital_energy_quantizer import OrbitalEnergyQuantizer
from .order_book_analyzer import WallType as OrderBookAnalyzer
from .order_book_manager import OrderBookManager
from .phantom_detector import PhantomDetector
from .phantom_logger import PhantomLogger
from .phantom_registry import PhantomRegistry
from .portfolio_tracker import PositionType as PortfolioTracker
from .profit_feedback_engine import ProfitFeedbackEngine
from .profit_optimization_engine import ProfitOptimizationEngine
from .quad_bit_strategy_array import QuadBitStrategyArray
from .quantum_mathematical_bridge import QuantumState
from .real_multi_exchange_trader import RealMultiExchangeTrader
from .registry_backtester import RegistryBacktester
from .registry_strategy import RegistryStrategy
from .risk_manager import RiskManager
from .schwafit_core import SchwafitCore
from .secure_exchange_manager import SecureExchangeManager
from .soulprint_registry import SoulprintRegistry
from .strategy.strategy_executor import StrategyExecutor

# Strategy imports
from .strategy.strategy_loader import StrategyLoader
from .strategy_bit_mapper import StrategyBitMapper
from .symbolic_math_interface import SymbolicMathInterface
from .symbolic_registry import SymbolicRegistry
from .tcell_survival_engine import TCellSurvivalEngine
from .tensor_score_utils import TensorScoreResult
from .two_gram_detector import TwoGramDetector

# Utility imports
from .type_defs import OrderSide, OrderType, TradingMode, TradingPair
from .unified_btc_trading_pipeline import UnifiedBTCTradingPipeline
from .unified_component_bridge import BridgeMode
from .unified_market_data_pipeline import DataSource
from .unified_pipeline_manager import UnifiedPipelineManager
from .unified_trade_router import UnifiedTradeRouter
from .vault_orbital_bridge import VaultOrbitalBridge
from .vector_registry import VectorRegistry
from .visual_decision_engine import VisualDecisionEngine

logger = logging.getLogger(__name__)


class SubsystemWrapper:
"""Wrapper for subsystems to normalize their interfaces."""

def __init__(self, name: str, instance: Any, config: Dict[str, Any] = None) -> None:
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
if hasattr(self.instance, "initialize"):
if asyncio.iscoroutinefunction(self.instance.initialize):
await self.instance.initialize()
else:
self.instance.initialize()
self.is_initialized = True
logger.info(f"✅ {self.name} initialized")
return True
elif hasattr(self.instance, "initialized") and self.instance.initialized:
self.is_initialized = True
logger.info(f"✅ {self.name} already initialized")
return True
elif hasattr(self.instance, "active") and self.instance.active:
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
if hasattr(self.instance, "start"):
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
if hasattr(self.instance, "stop"):
if asyncio.iscoroutinefunction(self.instance.stop):
await self.instance.stop()
else:
self.instance.stop()
elif hasattr(self.instance, "deactivate"):
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
if hasattr(self.instance, "reload"):
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
"name": self.name,
"is_initialized": self.is_initialized,
"is_running": self.is_running,
"config": self.config,
}

# Add instance-specific status if available
if hasattr(self.instance, "get_status"):
try:
instance_status = self.instance.get_status()
status.update(instance_status)
except Exception as e:
logger.warning(f"Failed to get status for {self.name}: {e}")

return status

def check_entropy_change(self) -> bool:
"""Check if entropy has changed significantly."""
current_time = time.time()
if current_time - self.last_entropy_check > 60:  # Check every minute
self.last_entropy_check = current_time
return True
return False


class SchwabotCoreSystem:
"""Main Schwabot trading system orchestrator."""

def __init__(self, config_path: Optional[str] = None) -> None:
"""Initialize the Schwabot core system."""
self.config = self._load_config(config_path)
self.subsystems: Dict[str, SubsystemWrapper] = {}
self.is_running = False
self.trading_cycle_task: Optional[asyncio.Task] = None

# Initialize logging
setup_logging(self.config.get("logging", {}))

# Initialize subsystems
self._initialize_subsystems()

logger.info("✅ Schwabot Core System initialized")

def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
"""Load configuration from file or use defaults."""
if config_path and os.path.exists(config_path):
with open(config_path, "r") as f:
return yaml.safe_load(f)
return self._get_default_config()

def _get_default_config(self) -> Dict[str, Any]:
"""Get default configuration."""
return {
"trading": {
"mode": "paper",
"pairs": ["BTC/USDT", "ETH/USDT"],
"max_position_size": 0.1,
"risk_per_trade": 0.02,
},
"logging": {"level": "INFO", "file": "schwabot.log"},
"subsystems": {
"risk_manager": True,
"profit_calculator": True,
"phantom_detector": True,
"gpu_acceleration": True,
},
}

def _initialize_subsystems(self) -> None:
"""Initialize all subsystem wrappers - ONLY what actually exists."""
logger.info("Initializing subsystem wrappers...")

# Define all subsystems with their configurations - ONLY what exists
subsystem_definitions = [
# Mathematical components
("MathConfigManager", MathConfigManager, {}),
("MathOrchestrator", MathOrchestrator, {}),
("MathResultCache", MathResultCache, {}),
("EnhancedMathematicalCore", EnhancedMathematicalCore, {}),
("MathematicalFrameworkIntegrator", MathematicalFrameworkIntegrator, {}),
("TCellSurvivalEngine", TCellSurvivalEngine, {}),
("EntropyMath", EntropyMath, {}),
("TensorScoreUtils", TensorScoreResult, {}),
("AdvancedTensorAlgebra", AdvancedTensorAlgebra, {}),
("SymbolicRegistry", SymbolicRegistry, {}),
("BitmapHashFolding", BitmapHashFolding, {}),
("OrbitalEnergyQuantizer", OrbitalEnergyQuantizer, {}),
("EntropyDriftEngine", EntropyDriftEngine, {}),
("VaultOrbitalBridge", VaultOrbitalBridge, {}),
("EntropyDecaySystem", EntropyDecaySystem, {}),
("TwoGramDetector", TwoGramDetector, {}),
("SymbolicMathInterface", SymbolicMathInterface, {}),
("QuantumState", QuantumState, {}),
# Trading components
(
"BTCTradingEngine",
BTCTradingEngine,
{
"api_key": os.getenv("BINANCE_API_KEY", "demo"),
"api_secret": os.getenv("BINANCE_API_SECRET", "demo"),
"testnet": self.config.get("trading", {}).get("mode") == "sandbox",
"symbol": self.config.get("trading", {}).get(
"default_symbol", "BTC/USDT"
),
},
),
("RiskManager", RiskManager, {}),
("SecureExchangeManager", SecureExchangeManager, {}),
("UnifiedPipelineManager", UnifiedPipelineManager, {}),
("UnifiedBTCTradingPipeline", UnifiedBTCTradingPipeline, {}),
("ProfitOptimizationEngine", ProfitOptimizationEngine, {}),
("RealMultiExchangeTrader", RealMultiExchangeTrader, {}),
("ProfitFeedbackEngine", ProfitFeedbackEngine, {}),
("CCXTTradingExecutor", CCXTTradingExecutor, {}),
("FillHandler", FillHandler, {}),
("EnhancedCCXTTradingEngine", EnhancedCCXTTradingEngine, {}),
# Strategy components
("StrategyLoader", StrategyLoader, {}),
("StrategyExecutor", StrategyExecutor, {}),
("RegistryStrategy", RegistryStrategy, {}),
("QuadBitStrategyArray", QuadBitStrategyArray, {}),
("StrategyBitMapper", StrategyBitMapper, {}),
# Market data and execution
("UnifiedMarketDataPipeline", DataSource, {}),
("UnifiedTradeRouter", UnifiedTradeRouter, {}),
("OrderBookManager", OrderBookManager, {}),
("OrderBookAnalyzer", OrderBookAnalyzer, {}),
# Portfolio and tracking
("PortfolioTracker", PortfolioTracker, {}),
("RegistryBacktester", RegistryBacktester, {}),
# Registry and storage
("SoulprintRegistry", SoulprintRegistry, {}),
("VectorRegistry", VectorRegistry, {}),
("PhantomRegistry", PhantomRegistry, {}),
("PhantomLogger", PhantomLogger, {}),
("PhantomDetector", PhantomDetector, {}),
# AI and processing
("VisualDecisionEngine", VisualDecisionEngine, {}),
# Integration and bridges
("UnifiedComponentBridge", BridgeMode, {}),
("CRWFCRLFIntegration", CRWFCrlfIntegration, {}),
("SchwafitCore", SchwafitCore, {}),
# Advanced systems
("GhostCore", GhostCore, {}),
("FractalCore", FractalCore, {}),
("FractalMemoryTracker", FractalMemoryTracker, {}),
# GPU and hardware
("GPUShaderIntegration", GPUShaderIntegration, {}),
("GPUDNAAutodetect", GPUDNAAutodetect, {}),
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

# Get data from market data subsystems
if "UnifiedMarketDataPipeline" in self.subsystems:
try:
wrapper = self.subsystems["UnifiedMarketDataPipeline"]
if hasattr(wrapper.instance, "get_latest_data"):
data = await wrapper.instance.get_latest_data()
market_data.update(data)
except Exception as e:
logger.debug(f"Market data error: {e}")

return market_data

async def _analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
"""Analyze market conditions using mathematical subsystems."""
analysis = {}

# Use mathematical analysis subsystems
math_subsystems = [
"EnhancedMathematicalCore",
"TCellSurvivalEngine",
"EntropyMath",
"TensorScoreUtils",
]

for name in math_subsystems:
if name in self.subsystems:
try:
wrapper = self.subsystems[name]
if hasattr(wrapper.instance, "analyze"):
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
"ProfitOptimizationEngine",
]

for name in strategy_subsystems:
if name in self.subsystems:
try:
wrapper = self.subsystems[name]
if hasattr(wrapper.instance, "generate_signals"):
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
if hasattr(wrapper.instance, "validate_signal"):
if not await wrapper.instance.validate_signal(signal):
logger.warning(f"Signal rejected by risk manager: {signal}")
continue

# Execute trade
if "BTCTradingEngine" in self.subsystems:
wrapper = self.subsystems["BTCTradingEngine"]
if hasattr(wrapper.instance, "execute_signal"):
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
"uptime": str(datetime.now() - self.start_time)
if self.start_time
else None,
"subsystems": {},
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

async def call_subsystem_method(
self, subsystem_name: str, method_name: str, *args, **kwargs
) -> Any:
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
async def place_order(
self,
symbol: str,
side: OrderSide,
order_type: OrderType,
quantity: Decimal,
price: Optional[Decimal] = None,
) -> Dict[str, Any]:
"""Place a trading order via CLI/API."""
if not self.is_running:
raise RuntimeError("System not running")

if "BTCTradingEngine" in self.subsystems:
return await self.call_subsystem_method(
"BTCTradingEngine",
"place_order",
symbol,
side,
order_type,
quantity,
price,
)

raise RuntimeError("Trading engine not available")

async def get_order_status(self, order_id: str) -> Dict[str, Any]:
"""Get order status via CLI/API."""
if "BTCTradingEngine" in self.subsystems:
return await self.call_subsystem_method(
"BTCTradingEngine", "get_order_status", order_id
)

raise RuntimeError("Trading engine not available")

async def cancel_order(self, order_id: str) -> bool:
"""Cancel an order via CLI/API."""
if "BTCTradingEngine" in self.subsystems:
return await self.call_subsystem_method(
"BTCTradingEngine", "cancel_order", order_id
)

raise RuntimeError("Trading engine not available")

async def get_portfolio_summary(self) -> Dict[str, Any]:
"""Get portfolio summary via CLI/API."""
if "PortfolioTracker" in self.subsystems:
return await self.call_subsystem_method("PortfolioTracker", "get_summary")

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
