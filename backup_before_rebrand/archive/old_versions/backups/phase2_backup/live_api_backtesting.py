"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live API Backtesting System for Schwabot.

This system connects to real exchanges, feeds live market data into the registry,
and enables actual trading when the system is turned on. This is NOT traditional
backtesting - it's live API integration for real-time trading with historical
data feeding into the decision-making process.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ccxt_trading_executor import CCXTTradingExecutor, IntegratedTradingSignal
from .clean_trading_pipeline import CleanTradingPipeline, create_trading_pipeline
from .portfolio_tracker import PortfolioTracker, create_portfolio_tracker
from .unified_market_data_pipeline import create_unified_pipeline

logger = logging.getLogger(__name__)


    class BacktestMode(Enum):
    """Class for Schwabot trading functionality."""
    """Class for Schwabot trading functionality."""
    """Backtesting modes."""

    LIVE_API = "live_api"  # Real API connections, live data
    SIMULATION = "simulation"  # Simulated trading with real data
    HISTORICAL = "historical"  # Historical data replay
    PAPER_TRADING = "paper_trading"  # Paper trading with live data


    @dataclass
        class LiveAPIConfig:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Configuration for live API backtesting."""

        exchange: str = "binance"
        api_key: str = ""
        api_secret: str = ""
        sandbox: bool = True
        symbols: List[str] = field(default_factory=lambda: ["BTC/USDC", "ETH/USDC"])
        update_interval: float = 1.0  # seconds
        enable_trading: bool = False
        max_position_size: float = 0.1
        risk_management: bool = True
        data_registry_path: str = "data/live_registry.json"
        portfolio_state_path: str = "data/portfolio_state.json"


        @dataclass
            class LiveMarketData:
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Live market data from API."""

            symbol: str
            price: Decimal
            volume: Decimal
            timestamp: float
            bid: Optional[Decimal] = None
            ask: Optional[Decimal] = None
            high_24h: Optional[Decimal] = None
            low_24h: Optional[Decimal] = None
            change_24h: Optional[Decimal] = None
            change_percent_24h: Optional[float] = None


            @dataclass
                class RegistryEntry:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Entry in the live data registry."""

                timestamp: float
                symbol: str
                price: float
                volume: float
                market_data: Dict[str, Any]
                trading_signals: Dict[str, Any]
                portfolio_state: Dict[str, Any]
                metadata: Dict[str, Any] = field(default_factory=dict)


                    class LiveAPIBacktesting:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """
                    Live API Backtesting System.

                        This system:
                        1. Connects to real exchanges via API
                        2. Feeds live market data into the registry
                        3. Enables actual trading when turned on
                        4. Maintains historical data for decision making
                        5. Provides real-time portfolio tracking
                        """

                            def __init__(self, config: LiveAPIConfig) -> None:
                            """Initialize live API backtesting system."""
                            self.config = config
                            self.is_running = False
                            self.is_trading_enabled = config.enable_trading

                            # Core components
                            self.trading_executor: Optional[CCXTTradingExecutor] = None
                            self.trading_pipeline: Optional[CleanTradingPipeline] = None
                            self.portfolio_tracker: Optional[PortfolioTracker] = None
                            self.market_data_pipeline = None

                            # Data storage
                            self.live_registry: List[RegistryEntry] = []
                            self.market_data_cache: Dict[str, LiveMarketData] = {}
                            self.signal_history: List[Dict[str, Any]] = []

                            # Performance tracking
                            self.start_time = time.time()
                            self.total_trades = 0
                            self.successful_trades = 0
                            self.total_pnl = Decimal("0")

                            # Ensure data directories exist
                            self._setup_directories()

                            logger.info("Live API Backtesting System initialized")

                                def _setup_directories(self) -> None:
                                """Setup necessary directories."""
                                Path("data").mkdir(exist_ok=True)
                                Path("logs").mkdir(exist_ok=True)
                                Path("portfolio").mkdir(exist_ok=True)

                                    async def initialize(self):
                                    """Initialize all components."""
                                        try:
                                        # Initialize trading executor
                                        executor_config = {
                                        "exchange": self.config.exchange,
                                        "apiKey": self.config.api_key,
                                        "secret": self.config.api_secret,
                                        "sandbox": self.config.sandbox,
                                        "simulation_mode": not self.is_trading_enabled,
                                        }

                                        self.trading_executor = CCXTTradingExecutor(executor_config)

                                        # Initialize portfolio tracker
                                        self.portfolio_tracker = create_portfolio_tracker()

                                        # Initialize trading pipeline
                                        self.trading_pipeline = create_trading_pipeline(
                                        symbol="BTCUSDC", initial_capital=10000.0, safe_mode=not self.is_trading_enabled
                                        )

                                        # Initialize market data pipeline
                                        self.market_data_pipeline = create_unified_pipeline()

                                        logger.info("All components initialized successfully")

                                            except Exception as e:
                                            logger.error(f"Failed to initialize components: {e}")
                                        raise

                                            async def start(self):
                                            """Start the live API backtesting system."""
                                                if self.is_running:
                                                logger.warning("System is already running")
                                            return

                                                try:
                                                await self.initialize()
                                                self.is_running = True

                                                logger.info("🚀 Live API Backtesting System STARTED")
                                                logger.info(f"Trading enabled: {self.is_trading_enabled}")
                                                logger.info(f"Exchange: {self.config.exchange}")
                                                logger.info(f"Symbols: {self.config.symbols}")

                                                # Start the main loop
                                                await self._main_loop()

                                                    except Exception as e:
                                                    logger.error(f"Failed to start system: {e}")
                                                    self.is_running = False
                                                raise

                                                    async def stop(self):
                                                    """Stop the live API backtesting system."""
                                                        if not self.is_running:
                                                        logger.warning("System is not running")
                                                    return

                                                    self.is_running = False

                                                    # Save final state
                                                    await self._save_final_state()

                                                    # Close connections
                                                        if self.trading_executor:
                                                        await self.trading_executor.close()

                                                        logger.info("🛑 Live API Backtesting System STOPPED")

                                                            async def _main_loop(self):
                                                            """Main processing loop."""
                                                                while self.is_running:
                                                                    try:
                                                                    # Fetch live market data
                                                                    market_data = await self._fetch_live_market_data()

                                                                    # Process market data through pipeline
                                                                        if market_data:
                                                                        await self._process_market_data(market_data)

                                                                        # Update portfolio and check positions
                                                                        await self._update_portfolio()

                                                                        # Save to registry
                                                                        await self._save_to_registry()

                                                                        # Wait for next update
                                                                        await asyncio.sleep(self.config.update_interval)

                                                                            except Exception as e:
                                                                            logger.error(f"Error in main loop: {e}")
                                                                            await asyncio.sleep(5)  # Wait before retrying

                                                                                async def _fetch_live_market_data(self) -> Optional[Dict[str, LiveMarketData]]:
                                                                                """Fetch live market data from exchange."""
                                                                                    if not self.trading_executor or not self.trading_executor.exchange:
                                                                                return None

                                                                                    try:
                                                                                    market_data = {}

                                                                                        for symbol in self.config.symbols:
                                                                                        # Fetch ticker data
                                                                                        ticker = await self.trading_executor.get_ticker(symbol)

                                                                                            if "error" not in ticker:
                                                                                            live_data = LiveMarketData(
                                                                                            symbol=symbol,
                                                                                            price=Decimal(str(ticker.get("last", 0))),
                                                                                            volume=Decimal(str(ticker.get("baseVolume", 0))),
                                                                                            timestamp=time.time(),
                                                                                            bid=Decimal(str(ticker.get("bid", 0))) if ticker.get("bid") else None,
                                                                                            ask=Decimal(str(ticker.get("ask", 0))) if ticker.get("ask") else None,
                                                                                            high_24h=(Decimal(str(ticker.get("high", 0))) if ticker.get("high") else None),
                                                                                            low_24h=Decimal(str(ticker.get("low", 0))) if ticker.get("low") else None,
                                                                                            change_24h=(Decimal(str(ticker.get("change", 0))) if ticker.get("change") else None),
                                                                                            change_percent_24h=(float(ticker.get("percentage", 0)) if ticker.get("percentage") else None),
                                                                                            )

                                                                                            market_data[symbol] = live_data
                                                                                            self.market_data_cache[symbol] = live_data

                                                                                        return market_data

                                                                                            except Exception as e:
                                                                                            logger.error(f"Failed to fetch market data: {e}")
                                                                                        return None

                                                                                            async def _process_market_data(self, market_data: Dict[str, LiveMarketData]):
                                                                                            """Process market data through trading pipeline."""
                                                                                                try:
                                                                                                    for symbol, data in market_data.items():
                                                                                                    # Convert to pipeline format
                                                                                                    pipeline_data = {
                                                                                                    "symbol": data.symbol,
                                                                                                    "price": float(data.price),
                                                                                                    "volume": float(data.volume),
                                                                                                    "timestamp": data.timestamp,
                                                                                                    "bid": float(data.bid) if data.bid else None,
                                                                                                    "ask": float(data.ask) if data.ask else None,
                                                                                                    "high_24h": float(data.high_24h) if data.high_24h else None,
                                                                                                    "low_24h": float(data.low_24h) if data.low_24h else None,
                                                                                                    "change_24h": float(data.change_24h) if data.change_24h else None,
                                                                                                    "change_percent_24h": data.change_percent_24h,
                                                                                                    }

                                                                                                    # Process through trading pipeline
                                                                                                        if self.trading_pipeline:
                                                                                                        result = await self.trading_pipeline.process_market_data(pipeline_data)

                                                                                                            if result and self.is_trading_enabled:
                                                                                                            # Generate trading signal
                                                                                                            signal = await self._generate_trading_signal(symbol, data, result)

                                                                                                                if signal:
                                                                                                                # Execute trade
                                                                                                                await self._execute_trade(signal)

                                                                                                                    except Exception as e:
                                                                                                                    logger.error(f"Failed to process market data: {e}")

                                                                                                                    async def _generate_trading_signal(
                                                                                                                    self, symbol: str, market_data: LiveMarketData, pipeline_result: Dict[str, Any]
                                                                                                                        ) -> Optional[IntegratedTradingSignal]:
                                                                                                                        """Generate trading signal from pipeline result."""
                                                                                                                            try:
                                                                                                                            # Extract trading decision from pipeline
                                                                                                                                if "trading_decision" not in pipeline_result:
                                                                                                                            return None

                                                                                                                            decision = pipeline_result["trading_decision"]

                                                                                                                                if decision.get("action") in ["buy", "sell"]:
                                                                                                                                signal = IntegratedTradingSignal(
                                                                                                                                signal_id=f"signal_{int(time.time())}",
                                                                                                                                recommended_action=decision["action"],
                                                                                                                                target_pair=symbol,
                                                                                                                                quantity=Decimal(str(decision.get("quantity", 0.001))),
                                                                                                                                confidence_score=Decimal(str(decision.get("confidence", 0.5))),
                                                                                                                                profit_potential=Decimal(str(decision.get("profit_potential", 0))),
                                                                                                                                risk_assessment=decision.get("risk_assessment", {}),
                                                                                                                                ghost_route=decision.get("strategy_branch", "unknown"),
                                                                                                                                )

                                                                                                                            return signal

                                                                                                                        return None

                                                                                                                            except Exception as e:
                                                                                                                            logger.error(f"Failed to generate trading signal: {e}")
                                                                                                                        return None

                                                                                                                            async def _execute_trade(self, signal: IntegratedTradingSignal):
                                                                                                                            """Execute a trading signal."""
                                                                                                                                try:
                                                                                                                                    if not self.trading_executor:
                                                                                                                                return

                                                                                                                                # Execute the signal
                                                                                                                                result = await self.trading_executor.execute_signal(signal)

                                                                                                                                    if result.executed:
                                                                                                                                    self.total_trades += 1

                                                                                                                                        if result.profit_realized and result.profit_realized > 0:
                                                                                                                                        self.successful_trades += 1

                                                                                                                                            if result.profit_realized:
                                                                                                                                            self.total_pnl += result.profit_realized

                                                                                                                                            logger.info(f"Trade executed: {signal.recommended_action} " f"{signal.quantity} {signal.target_pair}")
                                                                                                                                            logger.info(f"PnL: {result.profit_realized}")

                                                                                                                                            # Update portfolio tracker
                                                                                                                                                if self.portfolio_tracker:
                                                                                                                                                # This would update the portfolio based on the trade result
                                                                                                                                            pass

                                                                                                                                                except Exception as e:
                                                                                                                                                logger.error(f"Failed to execute trade: {e}")

                                                                                                                                                    async def _update_portfolio(self):
                                                                                                                                                    """Update portfolio state."""
                                                                                                                                                        try:
                                                                                                                                                            if self.portfolio_tracker:
                                                                                                                                                            # Update position prices
                                                                                                                                                            price_updates = {}
                                                                                                                                                                for symbol, data in self.market_data_cache.items():
                                                                                                                                                                price_updates[symbol] = data.price

                                                                                                                                                                self.portfolio_tracker.update_position_prices(price_updates)

                                                                                                                                                                # Check stop losses
                                                                                                                                                                positions_to_close = self.portfolio_tracker.check_stop_losses(price_updates)

                                                                                                                                                                    for position_id in positions_to_close:
                                                                                                                                                                        if self.market_data_cache.get(position_id):
                                                                                                                                                                        price = self.market_data_cache[position_id].price
                                                                                                                                                                        self.portfolio_tracker.close_position(position_id, price)
                                                                                                                                                                        logger.info(f"Position closed due to stop loss: {position_id}")

                                                                                                                                                                            except Exception as e:
                                                                                                                                                                            logger.error(f"Failed to update portfolio: {e}")

                                                                                                                                                                                async def _save_to_registry(self):
                                                                                                                                                                                """Save current state to registry."""
                                                                                                                                                                                    try:
                                                                                                                                                                                    # Create registry entry
                                                                                                                                                                                    entry = RegistryEntry(
                                                                                                                                                                                    timestamp=time.time(),
                                                                                                                                                                                    symbol="BTC/USDC",  # Primary symbol
                                                                                                                                                                                    price=float(
                                                                                                                                                                                    self.market_data_cache.get(
                                                                                                                                                                                    "BTC/USDC",
                                                                                                                                                                                    LiveMarketData("BTC/USDC", Decimal("0"), Decimal("0"), time.time()),
                                                                                                                                                                                    ).price
                                                                                                                                                                                    ),
                                                                                                                                                                                    volume=float(
                                                                                                                                                                                    self.market_data_cache.get(
                                                                                                                                                                                    "BTC/USDC",
                                                                                                                                                                                    LiveMarketData("BTC/USDC", Decimal("0"), Decimal("0"), time.time()),
                                                                                                                                                                                    ).volume
                                                                                                                                                                                    ),
                                                                                                                                                                                    market_data={
                                                                                                                                                                                    symbol: {
                                                                                                                                                                                    "price": float(data.price),
                                                                                                                                                                                    "volume": float(data.volume),
                                                                                                                                                                                    "timestamp": data.timestamp,
                                                                                                                                                                                    "bid": float(data.bid) if data.bid else None,
                                                                                                                                                                                    "ask": float(data.ask) if data.ask else None,
                                                                                                                                                                                    "high_24h": float(data.high_24h) if data.high_24h else None,
                                                                                                                                                                                    "low_24h": float(data.low_24h) if data.low_24h else None,
                                                                                                                                                                                    "change_24h": float(data.change_24h) if data.change_24h else None,
                                                                                                                                                                                    "change_percent_24h": data.change_percent_24h,
                                                                                                                                                                                    }
                                                                                                                                                                                    for symbol, data in self.market_data_cache.items()
                                                                                                                                                                                    },
                                                                                                                                                                                    trading_signals={
                                                                                                                                                                                    "total_trades": self.total_trades,
                                                                                                                                                                                    "successful_trades": self.successful_trades,
                                                                                                                                                                                    "total_pnl": float(self.total_pnl),
                                                                                                                                                                                    "is_trading_enabled": self.is_trading_enabled,
                                                                                                                                                                                    },
                                                                                                                                                                                    portfolio_state=(self.portfolio_tracker.get_portfolio_summary() if self.portfolio_tracker else {}),
                                                                                                                                                                                    )

                                                                                                                                                                                    self.live_registry.append(entry)

                                                                                                                                                                                    # Save to file periodically
                                                                                                                                                                                    if len(self.live_registry) % 100 == 0:  # Save every 100 entries
                                                                                                                                                                                    await self._save_registry_to_file()

                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                        logger.error(f"Failed to save to registry: {e}")

                                                                                                                                                                                            async def _save_registry_to_file(self):
                                                                                                                                                                                            """Save registry to file."""
                                                                                                                                                                                                try:
                                                                                                                                                                                                registry_data = [
                                                                                                                                                                                                {
                                                                                                                                                                                                "timestamp": entry.timestamp,
                                                                                                                                                                                                "symbol": entry.symbol,
                                                                                                                                                                                                "price": entry.price,
                                                                                                                                                                                                "volume": entry.volume,
                                                                                                                                                                                                "market_data": entry.market_data,
                                                                                                                                                                                                "trading_signals": entry.trading_signals,
                                                                                                                                                                                                "portfolio_state": entry.portfolio_state,
                                                                                                                                                                                                "metadata": entry.metadata,
                                                                                                                                                                                                }
                                                                                                                                                                                                for entry in self.live_registry
                                                                                                                                                                                                ]

                                                                                                                                                                                                    with open(self.config.data_registry_path, 'w') as f:
                                                                                                                                                                                                    json.dump(registry_data, f, indent=2)

                                                                                                                                                                                                    logger.debug(f"Registry saved with {len(self.live_registry)} entries")

                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                        logger.error(f"Failed to save registry to file: {e}")

                                                                                                                                                                                                            async def _save_final_state(self):
                                                                                                                                                                                                            """Save final state when stopping."""
                                                                                                                                                                                                                try:
                                                                                                                                                                                                                # Save registry
                                                                                                                                                                                                                await self._save_registry_to_file()

                                                                                                                                                                                                                # Save portfolio state
                                                                                                                                                                                                                    if self.portfolio_tracker:
                                                                                                                                                                                                                    self.portfolio_tracker.save_portfolio_state(self.config.portfolio_state_path)

                                                                                                                                                                                                                    # Save performance summary
                                                                                                                                                                                                                    performance_summary = {
                                                                                                                                                                                                                    "start_time": self.start_time,
                                                                                                                                                                                                                    "end_time": time.time(),
                                                                                                                                                                                                                    "total_trades": self.total_trades,
                                                                                                                                                                                                                    "successful_trades": self.successful_trades,
                                                                                                                                                                                                                    "success_rate": ((self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0),
                                                                                                                                                                                                                    "total_pnl": float(self.total_pnl),
                                                                                                                                                                                                                    "is_trading_enabled": self.is_trading_enabled,
                                                                                                                                                                                                                    "exchange": self.config.exchange,
                                                                                                                                                                                                                    "symbols": self.config.symbols,
                                                                                                                                                                                                                    }

                                                                                                                                                                                                                        with open("data/performance_summary.json", 'w') as f:
                                                                                                                                                                                                                        json.dump(performance_summary, f, indent=2)

                                                                                                                                                                                                                        logger.info("Final state saved")

                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                            logger.error(f"Failed to save final state: {e}")

                                                                                                                                                                                                                                def enable_trading(self) -> None:
                                                                                                                                                                                                                                """Enable live trading."""
                                                                                                                                                                                                                                self.is_trading_enabled = True
                                                                                                                                                                                                                                logger.info("🟢 LIVE TRADING ENABLED")

                                                                                                                                                                                                                                    def disable_trading(self) -> None:
                                                                                                                                                                                                                                    """Disable live trading."""
                                                                                                                                                                                                                                    self.is_trading_enabled = False
                                                                                                                                                                                                                                    logger.info("🔴 LIVE TRADING DISABLED")

                                                                                                                                                                                                                                        def get_status(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                        """Get current system status."""
                                                                                                                                                                                                                                    return {
                                                                                                                                                                                                                                    "is_running": self.is_running,
                                                                                                                                                                                                                                    "is_trading_enabled": self.is_trading_enabled,
                                                                                                                                                                                                                                    "total_trades": self.total_trades,
                                                                                                                                                                                                                                    "successful_trades": self.successful_trades,
                                                                                                                                                                                                                                    "success_rate": ((self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0),
                                                                                                                                                                                                                                    "total_pnl": float(self.total_pnl),
                                                                                                                                                                                                                                    "uptime": time.time() - self.start_time,
                                                                                                                                                                                                                                    "market_data_symbols": list(self.market_data_cache.keys()),
                                                                                                                                                                                                                                    "registry_entries": len(self.live_registry),
                                                                                                                                                                                                                                    "exchange": self.config.exchange,
                                                                                                                                                                                                                                    "symbols": self.config.symbols,
                                                                                                                                                                                                                                    }

                                                                                                                                                                                                                                        def get_portfolio_summary(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                        """Get portfolio summary."""
                                                                                                                                                                                                                                            if self.portfolio_tracker:
                                                                                                                                                                                                                                        return self.portfolio_tracker.get_portfolio_summary()
                                                                                                                                                                                                                                    return {}


                                                                                                                                                                                                                                        def create_live_api_backtesting(config: LiveAPIConfig) -> LiveAPIBacktesting:
                                                                                                                                                                                                                                        """Create a new live API backtesting instance."""
                                                                                                                                                                                                                                    return LiveAPIBacktesting(config)


                                                                                                                                                                                                                                        async def run_live_backtesting(config: LiveAPIConfig):
                                                                                                                                                                                                                                        """Run live API backtesting."""
                                                                                                                                                                                                                                        backtesting = create_live_api_backtesting(config)

                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                            await backtesting.start()
                                                                                                                                                                                                                                                except KeyboardInterrupt:
                                                                                                                                                                                                                                                logger.info("Received interrupt signal")
                                                                                                                                                                                                                                                    finally:
                                                                                                                                                                                                                                                    await backtesting.stop()


                                                                                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                                                                                        # Example configuration
                                                                                                                                                                                                                                                        config = LiveAPIConfig(
                                                                                                                                                                                                                                                        exchange="binance",
                                                                                                                                                                                                                                                        api_key="your_api_key",
                                                                                                                                                                                                                                                        api_secret="your_api_secret",
                                                                                                                                                                                                                                                        sandbox=True,
                                                                                                                                                                                                                                                        symbols=["BTC/USDC", "ETH/USDC"],
                                                                                                                                                                                                                                                        enable_trading=False,  # Start with trading disabled
                                                                                                                                                                                                                                                        update_interval=1.0,
                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                        # Run the system
                                                                                                                                                                                                                                                        asyncio.run(run_live_backtesting(config))
