"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Integration Manager
======================

The central coordinator for the Schwabot live API integration system.
Manages all exchange connections and provides a unified interface.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .data_models import APICredentials, MarketData, OrderRequest, OrderResponse, PortfolioPosition
from .enums import ConnectionStatus, ExchangeType
from .exchange_connection import ExchangeConnection

logger = logging.getLogger(__name__)


    class ApiIntegrationManager:
    """Class for Schwabot trading functionality."""
    """Class for Schwabot trading functionality."""
    """The main live API integration system."""

        def __init__(self, config_path: str = "config/api_keys.json") -> None:
        self.config_path = Path(config_path)
        self.connections: Dict[str, ExchangeConnection] = {}
        self.running = False
        self.market_data_feeds: Dict[str, asyncio.Task] = {}
        self.portfolio_positions: Dict[str, PortfolioPosition] = {}

        # Configuration from a separate config file could be used here
        self.update_interval = 5  # seconds
        self.heartbeat_interval = 30  # seconds
        self.reconnect_interval = 60  # seconds
        self.start_time = 0.0
        self.main_loop_task: Optional[asyncio.Task] = None

        self.load_configuration()
        logger.info("Live API Integration Manager initialized")

            def load_configuration(self) -> None:
            """Load API configurations from the specified JSON file."""
            logger.info("Loading API configuration from {0}...".format(self.config_path))

                try:
                    if not self.config_path.exists():
                    logger.warning(
                    "Configuration file not found: {0}. No exchanges will be loaded.".format(self.config_path)
                    )
                return

                    with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                        for exchange_name, exchange_config in config.items():
                            if exchange_config.get("enabled", True):
                                try:
                                credentials = APICredentials(
                                exchange=ExchangeType(exchange_name),
                                api_key=exchange_config.get("api_key"),
                                secret=exchange_config.get("secret"),
                            passphrase=exchange_config.get("passphrase"),
                            sandbox=exchange_config.get("sandbox", True),
                            testnet=exchange_config.get("testnet", True),
                            )

                            self.connections[exchange_name] = ExchangeConnection(credentials, exchange_config)

                                except (ValueError, KeyError) as e:
                                logger.error("Failed to parse config for '{0}': {1}".format(exchange_name, e))

                                logger.info("Loaded {0} exchange configurations.".format(len(self.connections)))

                                    except Exception as e:
                                    logger.error("Error loading API configuration: {0}".format(e), exc_info=True)

                                        async def start(self) -> None:
                                        """Start the API integration system and all connections."""
                                            if self.running:
                                            logger.warning("API integration manager is already running.")
                                        return

                                        self.running = True
                                        self.start_time = time.time()
                                        logger.info("Starting Live API Integration Manager...")

                                        await self._connect_all_exchanges()
                                        self.main_loop_task = asyncio.create_task(self._main_loop())

                                        # Market data feeds can be started here or on-demand
                                        logger.info("Live API Integration Manager started successfully.")

                                            async def stop(self) -> None:
                                            """Stop the API integration system gracefully."""
                                                if not self.running:
                                            return

                                            logger.info("Stopping Live API Integration Manager...")
                                            self.running = False

                                                if self.main_loop_task:
                                                self.main_loop_task.cancel()
                                                    try:
                                                    await self.main_loop_task
                                                        except asyncio.CancelledError:
                                                    pass  # Expected on cancellation

                                                        for task in self.market_data_feeds.values():
                                                        task.cancel()

                                                        await asyncio.gather(*(conn.disconnect() for conn in self.connections.values()))
                                                        logger.info("Live API Integration Manager stopped.")

                                                            async def _connect_all_exchanges(self) -> None:
                                                            """Attempt to connect to all loaded exchange configurations."""
                                                            connection_tasks = [conn.connect() for conn in self.connections.values()]
                                                            await asyncio.gather(*connection_tasks)

                                                                async def _main_loop(self) -> None:
                                                                """Run the main operational loop for health checks and portfolio updates."""
                                                                    while self.running:
                                                                        try:
                                                                        await self._heartbeat_check()
                                                                        await self._update_all_portfolios()
                                                                        await asyncio.sleep(self.update_interval)
                                                                            except asyncio.CancelledError:
                                                                        break
                                                                            except Exception as e:
                                                                            logger.error("Error in main loop: {0}".format(e), exc_info=True)
                                                                            await asyncio.sleep(self.reconnect_interval)

                                                                                async def _heartbeat_check(self) -> None:
                                                                                """Periodically check connection health and reconnect if necessary."""
                                                                                    for name, conn in self.connections.items():
                                                                                    if conn.status == ConnectionStatus.ERROR or (
                                                                                    conn.status == ConnectionStatus.CONNECTED
                                                                                    and time.time() - conn.last_heartbeat > self.heartbeat_interval
                                                                                        ):

                                                                                            if conn.status != ConnectionStatus.RECONNECTING:
                                                                                            logger.warning("Connection issue with {0}. Attempting to reconnect...".format(name))
                                                                                            conn.status = ConnectionStatus.RECONNECTING
                                                                                            conn.reconnect_attempts += 1

                                                                                                if conn.reconnect_attempts <= getattr(conn, 'max_reconnect_attempts', 5):
                                                                                                asyncio.create_task(conn.connect())
                                                                                                    else:
                                                                                                    logger.error("Max reconnect attempts reached for {0}. Disabling.".format(name))
                                                                                                    conn.status = ConnectionStatus.ERROR

                                                                                                        async def _update_all_portfolios(self) -> None:
                                                                                                        """Trigger portfolio updates for all connected exchanges."""
                                                                                                        # This can be expanded to fetch all balances and update a central portfolio model.
                                                                                                        # For now, it's a placeholder for periodic background tasks.
                                                                                                    pass

                                                                                                        async def place_order(self, exchange_name: str, order_request: OrderRequest) -> Optional[OrderResponse]:
                                                                                                        """Place an order on a specific exchange."""
                                                                                                        connection = self.connections.get(exchange_name)
                                                                                                            if connection and connection.status == ConnectionStatus.CONNECTED:
                                                                                                        return await connection.place_order(order_request)

                                                                                                        logger.error("Cannot place order: exchange '{0}' is not available.".format(exchange_name))
                                                                                                    return None

                                                                                                        async def get_market_data(self, exchange_name: str, symbol: str) -> Optional[MarketData]:
                                                                                                        """Get market data from a specific exchange."""
                                                                                                        connection = self.connections.get(exchange_name)
                                                                                                            if connection and connection.status == ConnectionStatus.CONNECTED:
                                                                                                        return await connection.get_market_data(symbol)

                                                                                                        logger.warning("Cannot get market data: exchange '{0}' is not available.".format(exchange_name))
                                                                                                    return None

                                                                                                        def get_system_status(self) -> Dict[str, Any]:
                                                                                                        """Provide a status overview of the entire API integration system."""
                                                                                                        uptime = time.time() - self.start_time if self.running else 0

                                                                                                    return {
                                                                                                    "running": self.running,
                                                                                                    "uptime_seconds": uptime,
                                                                                                    "uptime_formatted": str(timedelta(seconds=int(uptime))),
                                                                                                    "connections": {
                                                                                                    name: {
                                                                                                    "status": conn.status.value,
                                                                                                    "last_heartbeat": (
                                                                                                    datetime.fromtimestamp(conn.last_heartbeat).isoformat() if conn.last_heartbeat else None
                                                                                                    ),
                                                                                                    "reconnect_attempts": conn.reconnect_attempts,
                                                                                                    "last_error": conn.last_error,
                                                                                                    }
                                                                                                    for name, conn in self.connections.items()
                                                                                                    },
                                                                                                    }
