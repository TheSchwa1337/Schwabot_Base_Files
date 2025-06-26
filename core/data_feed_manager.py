# Import safe print for Windows compatibility
from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
try:
    pass
    pass
except ImportError:
    pass
    pass
    try:
    pass
    pass
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
    pass
    pass
def safe_print(message):


    pass
    pass
    print(message)
def info(message):


    pass
    pass
    print(f"[INFO] {message}")
def warn(message):


    pass
    pass
    print(f"[WARN] {message}")
def error(message):


    pass
    pass
    print(f"[ERROR] {message}")
def success(message):


    pass
    pass
    print(f"[SUCCESS] {message}")
def debug(message):


    pass
    pass
    print(f"[DEBUG] {message}")
# #!/usr/bin/env python3
"""
Data Feed Manager - Schwabot UROS v1.0
=======================================

Provides a unified interface for managing and consuming data from various
live and historical data feeds.

Features:
- Connection to multiple data sources (e.g., WebSocket, REST API, flat files)
- Normalization of data streams into a common format
- Management of feed lifecycles (start, stop, reconnect)
- Tick data processing and forwarding
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeedType(Enum):


    """Type of data feed."""
WEBSOCKET = "websocket"
REST_API = "rest_api"
CSV_FILE = "csv_file"
DATABASE = "database"


@dataclass
class FeedConfig:


    """Configuration for a single data feed."""
name: str
feed_type: FeedType
uri: str
symbol: str
update_interval: float = 1.0  # In seconds
is_active: bool = True
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TickData:


    """Standardized tick data format."""
symbol: str
price: float
volume: float
timestamp: datetime
source: str
metadata: Dict[str, Any] = field(default_factory=dict)


class DataFeedManager:


    """Manages multiple data feeds and provides a unified data stream."""

def __init__(self, configs: List[FeedConfig] = None):


    pass
    pass
        """Initialize the DataFeedManager."""
self.feeds: Dict[str, FeedConfig] = {}
self.active_tasks: Dict[str, asyncio.Task] = {}
self.subscribers: List[Callable[[TickData], None]] = []
        if configs:
            for config in configs:
self.add_feed(config)
        logger.info("DataFeedManager initialized.")

def add_feed(self, config: FeedConfig):


    pass
    pass
        """Add and configure a new data feed."""
        if config.name in self.feeds:
logger.warning(f"Feed '{config.name}' already exists. Overwriting.")
        self.feeds[config.name] = config
logger.info(f"Added data feed: {config.name} ({config.feed_type.value})")

def subscribe(self, callback: Callable[[TickData], None]):


    pass
    pass
        """Subscribe a callback function to receive tick data."""
        if callback not in self.subscribers:
self.subscribers.append(callback)
            logger.info(f"New subscriber added: {callback.__name__}")

async def start_all(self):
        """Start all active data feeds."""
logger.info("Starting all active data feeds...")
        for name, config in self.feeds.items():
            if config.is_active:
await self.start_feed(name)

async def start_feed(self, name: str):
        """Start a specific data feed."""
        if name not in self.feeds:
logger.error(f"Feed '{name}' not found.")
            return

        if name in self.active_tasks and not self.active_tasks[name].done():
            logger.warning(f"Feed '{name}' is already running.")
            return

config = self.feeds[name]
logger.info(f"Starting feed: {name}")
        task = asyncio.create_task(self._run_feed(config))
        self.active_tasks[name] = task

async def stop_all(self):
        """Stop all running data feeds."""
logger.info("Stopping all active data feeds...")
        for name in self.active_tasks:
await self.stop_feed(name)

async def stop_feed(self, name: str):
        """Stop a specific data feed."""
        if name not in self.active_tasks:
logger.warning(f"Feed '{name}' is not running.")
            return

task = self.active_tasks[name]
task.cancel()
        try:
    pass
    pass
await task
        except asyncio.CancelledError:
logger.info(f"Feed '{name}' stopped successfully.")
        del self.active_tasks[name]

async def _run_feed(self, config: FeedConfig):
        """The main loop for a single data feed."""
        # This is a placeholder for the actual feed implementation
        # A real implementation would connect to the source based on `feed_type`
logger.info(f"Running feed '{config.name}'...")
        while True:
            try:
    pass
    pass
                # Simulate fetching data
price = 100 + (hash(datetime.now()) % 10)
                volume = 1000 + (hash(datetime.now()) % 100)

tick = TickData(
                    symbol=config.symbol,
price=price,
volume=volume,
timestamp=datetime.now(),
                    source=config.name


                # Broadcast to subscribers
                for callback in self.subscribers:
callback(tick)

await asyncio.sleep(config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
logger.error(f"Error in feed '{config.name}': {e}")
                await asyncio.sleep(5)  # Wait before retrying

if __name__ == '__main__':
logging.basicConfig(level=logging.INFO)

    # Example of how to use the DataFeedManager
async def example_subscriber(tick: TickData):
        """A simple subscriber function to print received ticks."""
safe_print(f"Received tick from {tick.source}: {tick.symbol} - Price: ${tick.price:.2f}")

async def main():
        """Main function to demonstrate DataFeedManager."""
        # Configuration for two example feeds
feed_configs = [
FeedConfig(name="LiveBTC", feed_type=FeedType.WEBSOCKET, uri="wss://example.com/btc", symbol="BTC", update_interval=2),
            FeedConfig(name="HistoricalETH", feed_type=FeedType.CSV_FILE, uri="/data/eth.csv", symbol="ETH", update_interval=5),
        ]

manager = DataFeedManager(feed_configs)
        manager.subscribe(example_subscriber)

await manager.start_all()

        # Run for a short period
await asyncio.sleep(10)

await manager.stop_all()

asyncio.run(main())
