"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Loader - Load and manage trading strategies
===================================================

This module handles loading, validation, and management of trading strategies
for the Schwabot system.
"""

from ..type_defs import TradingStrategy
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import os
import logging
import inspect
import asyncio
import importlib


logger = logging.getLogger(__name__)

class StrategyLoader:
"""Class for Schwabot trading functionality."""
"""
Load and manage trading strategies.

This class handles loading strategies from files, validating them,
and providing access to them for the strategy executor.
"""


def __init__(self) -> None:
"""Initialize the strategy loader."""
self.strategies: Dict[str, TradingStrategy] = {}
self.strategy_paths: Dict[str, str] = {}
self.is_initialized = False
self.strategy_dir = Path("core/strategy")

logger.info("Strategy Loader initialized")

async def initialize(self) -> bool:
"""Initialize the strategy loader."""
try:
logger.info("Initializing Strategy Loader...")

# Create strategy directory if it doesn't exist
self.strategy_dir.mkdir(parents=True, exist_ok=True)

# Load all available strategies
await self.load_all_strategies()

self.is_initialized = True
logger.info(f"Strategy Loader initialized with {len(self.strategies)} strategies")
return True

except Exception as e:
logger.error(f"Failed to initialize Strategy Loader: {e}")
return False

async def load_all_strategies(self) -> bool:
"""Load all available strategies from the strategy directory."""
try:
logger.info("Loading all strategies...")

# Get all Python files in the strategy directory
strategy_files = list(self.strategy_dir.glob("*.py"))

for strategy_file in strategy_files:
if strategy_file.name.startswith("__"):
continue

try:
await self.load_strategy_from_file(strategy_file)
except Exception as e:
logger.error(f"Failed to load strategy from {strategy_file}: {e}")

logger.info(f"Loaded {len(self.strategies)} strategies")
return True

except Exception as e:
logger.error(f"Failed to load strategies: {e}")
return False

async def load_strategy_from_file(self, file_path: Path) -> Optional[TradingStrategy]:
"""Load a strategy from a Python file."""
try:
# Import the module
module_name = f"core.strategy.{file_path.stem}"
spec = importlib.util.spec_from_file_location(module_name, file_path)

if spec is None or spec.loader is None:
logger.error(f"Could not load spec for {file_path}")
return None

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Find strategy classes in the module
strategy_classes = []
for name, obj in inspect.getmembers(module):
if (inspect.isclass(obj) and
issubclass(obj, TradingStrategy) and
obj != TradingStrategy):
strategy_classes.append(obj)

if not strategy_classes:
logger.warning(f"No strategy classes found in {file_path}")
return None

# Load the first strategy class found:
strategy_class = strategy_classes[0]
strategy_name = strategy_class.__name__

# Create strategy instance
strategy = strategy_class()

# Validate strategy
if not await self.validate_strategy(strategy):
logger.error(f"Strategy validation failed for {strategy_name}")
return None

# Store strategy
self.strategies[strategy_name] = strategy
self.strategy_paths[strategy_name] = str(file_path)

logger.info(f"Loaded strategy: {strategy_name}")
return strategy

except Exception as e:
logger.error(f"Failed to load strategy from {file_path}: {e}")
return None

async def validate_strategy(self, strategy: TradingStrategy) -> bool:
"""Validate a trading strategy."""
try:
# Check required methods
required_methods = [
'initialize',
'analyze',
'generate_signals',
'get_name',
'get_description'
]

for method_name in required_methods:
if not hasattr(strategy, method_name):
logger.error(f"Strategy missing required method: {method_name}")
return False

# Check if strategy can be initialized
if hasattr(strategy, 'initialize'):
await strategy.initialize()

logger.info(
f"Strategy validation passed: {
strategy.get_name()}")
return True

except Exception as e:
logger.error(f"Strategy validation failed: {e}")
return False

def get_strategy(
self, strategy_name: str) -> Optional[TradingStrategy]:
"""Get a strategy by name."""
return self.strategies.get(strategy_name)

def get_all_strategies(self) -> Dict[str, TradingStrategy]:
"""Get all loaded strategies."""
return self.strategies.copy()

def get_strategy_names(self) -> List[str]:
"""Get names of all loaded strategies."""
return list(self.strategies.keys())

async def reload_strategy(self, strategy_name: str) -> bool:
"""Reload a specific strategy."""
try:
if strategy_name not in self.strategy_paths:
logger.error(f"Strategy not found: {strategy_name}")
return False

file_path = Path(self.strategy_paths[strategy_name])

# Remove old strategy
if strategy_name in self.strategies:
del self.strategies[strategy_name]

# Load new strategy
strategy = await self.load_strategy_from_file(file_path)

if strategy is None:
logger.error(
f"Failed to reload strategy: {strategy_name}")
return False

logger.info(f"Reloaded strategy: {strategy_name}")
return True

except Exception as e:
logger.error(
f"Failed to reload strategy {strategy_name}: {e}")
return False

async def reload_all_strategies(self) -> bool:
"""Reload all strategies."""
try:
logger.info("Reloading all strategies...")

# Clear existing strategies
self.strategies.clear()
self.strategy_paths.clear()

# Reload all strategies
success = await self.load_all_strategies()

if success:
logger.info(
"All strategies reloaded successfully")
else:
logger.error(
"Failed to reload all strategies")

return success

except Exception as e:
logger.error(
f"Failed to reload all strategies: {e}")
return False

def get_strategy_info(
self, strategy_name: str) -> Optional[Dict[str, Any]]:
"""Get information about a strategy."""
strategy = self.get_strategy(strategy_name)
if strategy is None:
return None

return {
"name": strategy.get_name(),
"description": strategy.get_description(),
"file_path": self.strategy_paths.get(strategy_name),
"methods": [method for method in dir(strategy) if not method.startswith("_")],
"is_initialized": hasattr(strategy, 'is_initialized') and strategy.is_initialized
}

def get_all_strategy_info(
self) -> Dict[str, Dict[str, Any]]:
"""Get information about all strategies."""
info = {}
for strategy_name in self.strategies.keys():
info[strategy_name] = self.get_strategy_info(
strategy_name)
return info

async def cleanup(self):
"""Clean up resources."""
try:
logger.info(
"Cleaning up Strategy Loader...")

# Clean up all strategies
for strategy in self.strategies.values():
if hasattr(
strategy, 'cleanup'):
await strategy.cleanup()

self.strategies.clear()
self.strategy_paths.clear()

logger.info(
"Strategy Loader cleanup completed")

except Exception as e:
logger.error(
f"Error during Strategy Loader cleanup: {e}")

# Example strategy class
# for testing
class ExampleStrategy(
TradingStrategy):
"""Class for Schwabot trading functionality."""
"""Example trading strategy for testing."""

def __init__(self) -> None:
super().__init__()
self.name = "Example Strategy"
self.description = "An example trading strategy for testing purposes"

async def initialize(self) -> bool:
"""Initialize the strategy."""
logger.info("Initializing Example Strategy")
self.is_initialized = True
return True

async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
"""Analyze market data."""
return {
"signal_strength": 0.5,
"confidence": 0.7,
"recommendation": "HOLD"
}

async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
"""Generate trading signals."""
signals = []

if analysis.get("signal_strength", 0) > 0.6:
signals.append({
"type": "BUY",
"symbol": "BTC/USDT",
"quantity": 0.001,
"confidence": analysis.get("confidence", 0)
})

return signals

def get_name(self) -> str:
"""Get strategy name."""
return self.name

def get_description(self) -> str:
"""Get strategy description."""
return self.description