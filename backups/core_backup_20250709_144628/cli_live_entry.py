"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 CLI Live Entry Point for Schwabot Trading System
==================================================

    Production CLI interface that integrates:
    - 2-gram pattern detection for strategy routing
    - Real-time market data processing
    - CCXT trading execution across multiple exchanges
    - Strategy trigger routing with dual-state optimization
    - Portfolio balancing and risk management
    - Live performance monitoring and logging

    This is the main entry point for live trading operations.
    """

    import argparse
    import asyncio
    import json
    import logging
    import os  # Added for os.getenv
    import signal
    import sys
    import time
    from pathlib import Path
    from typing import Any, Dict, List, Optional

    # Configuration and utilities
    from config.schwabot_config import load_config
    from core.algorithmic_portfolio_balancer import AlgorithmicPortfolioBalancer, create_portfolio_balancer
    from core.btc_usdc_trading_integration import BTCUSDCTradingIntegration, create_btc_usdc_integration

    # New trading components
    from core.ccxt_trading_executor import CCXTTradingExecutor, create_ccxt_trading_executor
    from core.clean_trading_pipeline import CleanTradingPipeline, create_trading_pipeline
    from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
    from core.master_profit_coordination_system import (
        MasterProfitCoordinationSystem,
        create_master_profit_coordination_system,
    )
    from core.order_book_manager import OrderBookManager, create_order_book_manager
    from core.real_time_execution_engine import RealTimeExecutionEngine, create_real_time_execution_engine
    from core.strategy_trigger_router import StrategyTriggerRouter, create_strategy_trigger_router
    from core.trading_strategy_executor import TradingStrategyExecutor, create_trading_strategy_executor

    # Core Schwabot imports
    from core.two_gram_detector import TwoGramDetector, create_two_gram_detector
    from utils.logging_setup import setup_logging

    logger = logging.getLogger(__name__)


        class SchwabotCLI:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """
        Main CLI interface for Schwabot trading system.

            Handles:
            - System initialization and component coordination
            - Live market data processing with 2-gram detection
            - Strategy routing and execution
            - Portfolio management and risk control
            - Performance monitoring and reporting
            """

                def __init__(self, config_path: str = "config/schwabot_config.yaml") -> None:
                """Initialize the Schwabot CLI system."""
                self.config_path = config_path
                self.config = None

                # Core components
                self.two_gram_detector: Optional[TwoGramDetector] = None
                self.strategy_router: Optional[StrategyTriggerRouter] = None
                self.trading_pipeline: Optional[CleanTradingPipeline] = None
                self.execution_engine: Optional[RealTimeExecutionEngine] = None
                self.entropy_executor: Optional[EntropyEnhancedTradingExecutor] = None
                self.portfolio_balancer: Optional[AlgorithmicPortfolioBalancer] = None
                self.btc_integration: Optional[BTCUSDCTradingIntegration] = None
                self.master_system: Optional[MasterProfitCoordinationSystem] = None

                # New trading components
                self.ccxt_executor: Optional[CCXTTradingExecutor] = None
                self.order_book_manager: Optional[OrderBookManager] = None
                self.trading_strategy_executor: Optional[TradingStrategyExecutor] = None

                # System state
                self.running = False
                self.mode = "demo"  # demo, live, backtest
                self.exchanges = []  # List of configured exchanges
                self.active_pairs = []  # List of active trading pairs

                # Performance tracking
                self.start_time = None
                self.total_trades = 0
                self.total_profit = 0.0
                self.current_positions = {}

                # Control flags
                self.shutdown_requested = False

                logger.info("🧠 Schwabot CLI initialized")

                    async def initialize_system(self, mode: str = "demo") -> bool:
                    """Initialize all system components."""
                        try:
                        logger.info("🔧 Initializing Schwabot trading system...")

                        # Load configuration
                        self.config = load_config(self.config_path)
                        self.mode = mode

                        # Auto-detect portfolio and assets from exchanges
                        await self._discover_portfolio_and_assets()

                        # Initialize 2-gram detector (foundation layer)
                        logger.info("🧬 Initializing 2-gram detector...")
                        self.two_gram_detector = create_two_gram_detector(self.config.get("2gram_config", {}))

                        # Initialize strategy trigger router
                        logger.info("🎯 Initializing strategy trigger router...")
                        self.strategy_router = create_strategy_trigger_router(self.config.get("strategy_router_config", {}))

                        # Initialize trading pipeline with discovered portfolio
                        logger.info("🔄 Initializing trading pipeline...")
                        self.trading_pipeline = create_trading_pipeline(
                        symbol=self.config.get("default_symbol", "BTC/USDC"),
                        initial_capital=self.total_portfolio_value,  # Use discovered value
                        safe_mode=(mode == "demo"),
                        )

                        # Initialize real-time execution engine
                        logger.info("⚡ Initializing real-time execution engine...")
                        self.execution_engine = create_real_time_execution_engine(self.config.get("execution_engine_config", {}))

                        # Initialize entropy-enhanced trading executor
                        logger.info("🧠 Initializing entropy-enhanced trading executor...")
                        self.entropy_executor = EntropyEnhancedTradingExecutor(
                        config=self.config.get("entropy_executor_config", {})
                        )

                        # Initialize portfolio balancer with discovered assets
                        logger.info("⚖️ Initializing portfolio balancer...")
                        self.portfolio_balancer = create_portfolio_balancer(self.config.get("portfolio_balancer_config", {}))

                        # Initialize BTC/USDC integration
                        logger.info("💰 Initializing BTC/USDC integration...")
                        self.btc_integration = create_btc_usdc_integration(self.config.get("btc_usdc_config", {}))

                        # Initialize master profit coordination system
                        logger.info("🎛️ Initializing master profit coordination system...")
                        self.master_system = create_master_profit_coordination_system(self.config.get("master_system_config", {}))

                        # Initialize new trading components
                        logger.info("💼 Initializing CCXT trading executor...")
                        self.ccxt_executor = create_ccxt_trading_executor(self.config.get("ccxt_executor_config", {}))

                        logger.info("📊 Initializing order book manager...")
                        self.order_book_manager = create_order_book_manager(self.config.get("order_book_manager_config", {}))

                        logger.info("🎯 Initializing trading strategy executor...")
                        self.trading_strategy_executor = create_trading_strategy_executor(
                        self.config.get("trading_strategy_executor_config", {})
                        )

                        # Initialize trading components
                        await self._initialize_trading_components()

                        # Inject components into master system
                        await self._inject_components()

                        # Initialize execution engine
                        await self.execution_engine.initialize()

                        logger.info("✅ Schwabot system initialization completed")
                    return True

                        except Exception as e:
                        logger.error(f"❌ System initialization failed: {e}")
                    return False

                        async def _discover_portfolio_and_assets(self):
                        """Auto-detect portfolio metrics and held assets from exchange APIs."""
                            try:
                            logger.info("🔍 Discovering portfolio and assets from exchanges...")

                            # Initialize portfolio tracking
                            self.total_portfolio_value = 0.0
                            self.held_assets = {}
                            self.available_balances = {}
                            self.active_pairs = []

                            # Get configured exchanges
                            exchanges = self.config.get("exchanges", [])

                                for exchange_config in exchanges:
                                    if not exchange_config.get("enabled", False):
                                continue

                                exchange_name = exchange_config.get("name", "unknown")
                                logger.info(f"📊 Discovering portfolio from {exchange_name}...")

                                    try:
                                    # Get exchange connection
                                    exchange_connection = await self._get_exchange_connection(exchange_config)

                                        if exchange_connection:
                                        # Fetch account balance
                                        balance = await self._fetch_account_balance(exchange_connection, exchange_name)

                                        # Fetch held assets
                                        assets = await self._fetch_held_assets(exchange_connection, exchange_name)

                                        # Fetch available trading pairs
                                        pairs = await self._fetch_trading_pairs(exchange_connection, exchange_name)

                                        # Update portfolio metrics
                                        self._update_portfolio_metrics(exchange_name, balance, assets, pairs)

                                        logger.info(
                                        f"✅ {exchange_name}: ${balance['total_value']:.2f} total, " f"{len(assets)} assets"
                                        )

                                            except Exception as e:
                                            logger.warning(f"⚠️ Failed to discover portfolio from {exchange_name}: {e}")
                                        continue

                                        # Log discovered portfolio
                                        logger.info("📊 Portfolio Discovery Complete:")
                                        logger.info(f"   Total Portfolio Value: ${self.total_portfolio_value:.2f}")
                                        logger.info(f"   Total Assets: {len(self.held_assets)}")
                                        logger.info(f"   Available Pairs: {len(self.active_pairs)}")

                                        # Log held assets
                                            for asset, amount in self.held_assets.items():
                                                if amount > 0:
                                                logger.info(f"     {asset}: {amount}")

                                                # Update configuration with discovered values
                                                self._update_config_with_discovered_values()

                                                    except Exception as e:
                                                    logger.error(f"❌ Portfolio discovery failed: {e}")
                                                    # Fallback to demo values
                                                    self.total_portfolio_value = 10000.0
                                                    self.held_assets = {"USDC": 10000.0}
                                                    self.active_pairs = ["BTC/USDC", "ETH/USDC"]

                                                        async def _get_exchange_connection(self, exchange_config):
                                                        """Get connection to exchange."""
                                                            try:
                                                            exchange_name = exchange_config.get("name", "unknown")

                                                                if self.mode == "demo":
                                                                # Return mock connection for demo mode
                                                            return {"name": exchange_name, "mode": "demo"}

                                                            # Get API credentials from environment
                                                            api_key_env = exchange_config.get("api_key_env")
                                                            secret_env = exchange_config.get("secret_env")

                                                                if not api_key_env or not secret_env:
                                                                logger.warning(f"⚠️ Missing API credentials for {exchange_name}")
                                                            return None

                                                            api_key = os.getenv(api_key_env)
                                                            secret = os.getenv(secret_env)

                                                                if not api_key or not secret:
                                                                logger.warning(f"⚠️ API credentials not set for {exchange_name}")
                                                            return None

                                                            # Create exchange connection
                                                            # This would integrate with your existing exchange connection logic
                                                            from core.api.exchange_connection import (
                                                                create_exchange_connection,
                                                            )

                                                            connection = await create_exchange_connection(
                                                            exchange_name=exchange_name,
                                                            api_key=api_key,
                                                            secret=secret,
                                                            sandbox=exchange_config.get("sandbox", False),
                                                            )

                                                        return connection

                                                            except Exception as e:
                                                            logger.error(f"Error creating exchange connection: {e}")
                                                        return None

                                                            async def _fetch_account_balance(self, connection, exchange_name):
                                                            """Fetch account balance from exchange."""
                                                                try:
                                                                    if connection.get("mode") == "demo":
                                                                    # Mock balance for demo mode
                                                                return {
                                                                "total_value": 10000.0,
                                                                "currencies": {
                                                                "USDC": {"free": 10000.0, "used": 0.0, "total": 10000.0},
                                                                "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                                                                "ETH": {"free": 0.0, "used": 0.0, "total": 0.0},
                                                                },
                                                                }

                                                                # Fetch real balance from exchange
                                                                balance = await connection.fetch_balance()

                                                                # Calculate total value in USD
                                                                total_value = 0.0
                                                                    for currency, amounts in balance.items():
                                                                        if amounts['total'] > 0:
                                                                        # Get current price for currency
                                                                            try:
                                                                                if currency != 'USDC' and currency != 'USD':
                                                                                ticker = await connection.fetch_ticker(f"{currency}/USDC")
                                                                                price = ticker['last']
                                                                                total_value += amounts['total'] * price
                                                                                    else:
                                                                                    total_value += amounts['total']
                                                                                        except Exception:
                                                                                        # If can't get price, assume it's worth something
                                                                                        total_value += amounts['total']

                                                                                    return {"total_value": total_value, "currencies": balance}

                                                                                        except Exception as e:
                                                                                        logger.error(f"Error fetching balance from {exchange_name}: {e}")
                                                                                    return {"total_value": 0.0, "currencies": {}}

                                                                                        async def _fetch_held_assets(self, connection, exchange_name):
                                                                                        """Fetch held assets from exchange."""
                                                                                            try:
                                                                                                if connection.get("mode") == "demo":
                                                                                                # Mock assets for demo mode
                                                                                            return {"USDC": 10000.0, "BTC": 0.0, "ETH": 0.0}

                                                                                            # Fetch real assets from exchange
                                                                                            balance = await connection.fetch_balance()

                                                                                            assets = {}
                                                                                                for currency, amounts in balance.items():
                                                                                                    if amounts['total'] > 0:
                                                                                                    assets[currency] = amounts['total']

                                                                                                return assets

                                                                                                    except Exception as e:
                                                                                                    logger.error(f"Error fetching assets from {exchange_name}: {e}")
                                                                                                return {}

                                                                                                    async def _fetch_trading_pairs(self, connection, exchange_name):
                                                                                                    """Fetch available trading pairs from exchange."""
                                                                                                        try:
                                                                                                            if connection.get("mode") == "demo":
                                                                                                            # Mock pairs for demo mode
                                                                                                        return ["BTC/USDC", "ETH/USDC", "BTC/USDT", "ETH/USDT"]

                                                                                                        # Fetch real trading pairs from exchange
                                                                                                        markets = await connection.fetch_markets()

                                                                                                        pairs = []
                                                                                                            for market in markets:
                                                                                                                if market['active'] and market['spot']:
                                                                                                                pairs.append(market['symbol'])

                                                                                                            return pairs

                                                                                                                except Exception as e:
                                                                                                                logger.error(f"Error fetching pairs from {exchange_name}: {e}")
                                                                                                            return []

                                                                                                                def _update_portfolio_metrics(self, exchange_name, balance, assets, pairs) -> None:
                                                                                                                """Update portfolio metrics with discovered data."""
                                                                                                                    try:
                                                                                                                    # Update total portfolio value
                                                                                                                    self.total_portfolio_value += balance.get("total_value", 0.0)

                                                                                                                    # Update held assets
                                                                                                                        for asset, amount in assets.items():
                                                                                                                            if asset in self.held_assets:
                                                                                                                            self.held_assets[asset] += amount
                                                                                                                                else:
                                                                                                                                self.held_assets[asset] = amount

                                                                                                                                # Update available balances
                                                                                                                                self.available_balances[exchange_name] = balance.get("currencies", {})

                                                                                                                                # Update active pairs
                                                                                                                                    for pair in pairs:
                                                                                                                                        if pair not in self.active_pairs:
                                                                                                                                        self.active_pairs.append(pair)

                                                                                                                                            except Exception as e:
                                                                                                                                            logger.error(f"Error updating portfolio metrics: {e}")

                                                                                                                                                def _update_config_with_discovered_values(self) -> None:
                                                                                                                                                """Update configuration with discovered portfolio values."""
                                                                                                                                                    try:
                                                                                                                                                    # Update initial capital with discovered portfolio value
                                                                                                                                                        if self.total_portfolio_value > 0:
                                                                                                                                                        self.config["initial_capital"] = self.total_portfolio_value
                                                                                                                                                        logger.info(f"💰 Updated initial capital to: ${self.total_portfolio_value:.2f}")

                                                                                                                                                        # Update default symbol if we have assets
                                                                                                                                                            if self.held_assets:
                                                                                                                                                            # Find the most liquid pair
                                                                                                                                                                if "BTC" in self.held_assets and "USDC" in self.held_assets:
                                                                                                                                                                self.config["default_symbol"] = "BTC/USDC"
                                                                                                                                                                    elif "ETH" in self.held_assets and "USDC" in self.held_assets:
                                                                                                                                                                    self.config["default_symbol"] = "ETH/USDC"
                                                                                                                                                                        elif "BTC" in self.active_pairs:
                                                                                                                                                                        self.config["default_symbol"] = "BTC/USDC"
                                                                                                                                                                        logger.info(f"🎯 Updated default symbol to: {self.config['default_symbol']}")
                                                                                                                                                                        # Update portfolio balancer target allocation based on held assets
                                                                                                                                                                            if self.held_assets:
                                                                                                                                                                            total_value = sum(self.held_assets.values())
                                                                                                                                                                                if total_value > 0:
                                                                                                                                                                                target_allocation = {}
                                                                                                                                                                                    for asset, amount in self.held_assets.items():
                                                                                                                                                                                        if amount > 0:
                                                                                                                                                                                        target_allocation[asset] = amount / total_value
                                                                                                                                                                                        # Update portfolio balancer config
                                                                                                                                                                                            if "portfolio_balancer_config" not in self.config:
                                                                                                                                                                                            self.config["portfolio_balancer_config"] = {}
                                                                                                                                                                                            self.config["portfolio_balancer_config"]["target_allocation"] = target_allocation
                                                                                                                                                                                            logger.info("⚖️ Updated target allocation based on held assets")
                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                logger.error(f"Error updating config with discovered values: {e}")

                                                                                                                                                                                                    async def _initialize_trading_components(self):
                                                                                                                                                                                                    """Initialize all new trading components."""
                                                                                                                                                                                                        try:
                                                                                                                                                                                                        # Inject new components into master system
                                                                                                                                                                                                        await self.master_system.inject_trading_components(
                                                                                                                                                                                                        ccxt_executor=self.ccxt_executor,
                                                                                                                                                                                                        order_book_manager=self.order_book_manager,
                                                                                                                                                                                                        trading_strategy_executor=self.trading_strategy_executor,
                                                                                                                                                                                                        )
                                                                                                                                                                                                        # Inject components into strategy router
                                                                                                                                                                                                        self.strategy_router.ccxt_executor = self.ccxt_executor
                                                                                                                                                                                                        self.strategy_router.order_book_manager = self.order_book_manager
                                                                                                                                                                                                        self.strategy_router.trading_strategy_executor = self.trading_strategy_executor
                                                                                                                                                                                                        logger.info("🔗 New component injection completed")
                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                            logger.error(f"❌ New component injection failed: {e}")
                                                                                                                                                                                                        raise

                                                                                                                                                                                                            async def _inject_components(self):
                                                                                                                                                                                                            """Inject all components into the master coordination system."""
                                                                                                                                                                                                                try:
                                                                                                                                                                                                                # Inject trading components into master system
                                                                                                                                                                                                                await self.master_system.inject_trading_components(
                                                                                                                                                                                                                two_gram_detector=self.two_gram_detector,
                                                                                                                                                                                                                strategy_router=self.strategy_router,
                                                                                                                                                                                                                trading_pipeline=self.trading_pipeline,
                                                                                                                                                                                                                execution_engine=self.execution_engine,
                                                                                                                                                                                                                entropy_executor=self.entropy_executor,
                                                                                                                                                                                                                portfolio_balancer=self.portfolio_balancer,
                                                                                                                                                                                                                btc_integration=self.btc_integration,
                                                                                                                                                                                                                ccxt_executor=self.ccxt_executor,
                                                                                                                                                                                                                order_book_manager=self.order_book_manager,
                                                                                                                                                                                                                trading_strategy_executor=self.trading_strategy_executor,
                                                                                                                                                                                                                )
                                                                                                                                                                                                                # Inject components into strategy router
                                                                                                                                                                                                                self.strategy_router.portfolio_balancer = self.portfolio_balancer
                                                                                                                                                                                                                self.strategy_router.btc_usdc_integration = self.btc_integration
                                                                                                                                                                                                                logger.info("🔗 Component injection completed")
                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                    logger.error(f"❌ Component injection failed: {e}")
                                                                                                                                                                                                                raise

                                                                                                                                                                                                                    async def start_live_trading(self) -> bool:
                                                                                                                                                                                                                    """Start live trading operations."""
                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                        logger.info("🚀 Starting live trading operations...")
                                                                                                                                                                                                                            if self.mode == "demo":
                                                                                                                                                                                                                            logger.info("🎮 Running in DEMO mode - no real trades will be executed")
                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                logger.info("💼 Running in LIVE mode - real trades will be executed")
                                                                                                                                                                                                                                self.running = True
                                                                                                                                                                                                                                self.start_time = time.time()
                                                                                                                                                                                                                                # Start execution engine
                                                                                                                                                                                                                                await self.execution_engine.start_monitoring()
                                                                                                                                                                                                                                # Start market monitoring task
                                                                                                                                                                                                                                asyncio.create_task(self._market_monitoring_loop())
                                                                                                                                                                                                                                # Start strategy processing task
                                                                                                                                                                                                                                asyncio.create_task(self._strategy_processing_loop())
                                                                                                                                                                                                                                # Start performance monitoring task
                                                                                                                                                                                                                                asyncio.create_task(self._performance_monitoring_loop())
                                                                                                                                                                                                                                logger.info("✅ Live trading started successfully")
                                                                                                                                                                                                                            return True
                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                logger.error(f"❌ Failed to start live trading: {e}")
                                                                                                                                                                                                                            return False

                                                                                                                                                                                                                                async def _market_monitoring_loop(self):
                                                                                                                                                                                                                                """Main market monitoring loop with 2-gram detection."""
                                                                                                                                                                                                                                logger.info("📊 Starting market monitoring loop...")
                                                                                                                                                                                                                                    while self.running and not self.shutdown_requested:
                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                        # Get market data from execution engine
                                                                                                                                                                                                                                        market_data = await self.execution_engine._get_current_market_state()
                                                                                                                                                                                                                                            if market_data:
                                                                                                                                                                                                                                            # Process market data through 2-gram detector
                                                                                                                                                                                                                                            await self._process_market_data_with_2gram(market_data)
                                                                                                                                                                                                                                            # Wait for next cycle
                                                                                                                                                                                                                                            await asyncio.sleep(self.config.get("market_monitoring_interval", 1.0))
                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                logger.error(f"Error in market monitoring loop: {e}")
                                                                                                                                                                                                                                                await asyncio.sleep(5.0)  # Wait before retrying

                                                                                                                                                                                                                                                    async def _process_market_data_with_2gram(self, market_data: Dict[str, Any]):
                                                                                                                                                                                                                                                    """Process market data through 2-gram detection and strategy routing."""
                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                        # Convert market data to sequence for 2-gram analysis
                                                                                                                                                                                                                                                        market_sequence = self._convert_market_data_to_sequence(market_data)

                                                                                                                                                                                                                                                            if len(market_sequence) >= 2:
                                                                                                                                                                                                                                                            # Analyze with 2-gram detector
                                                                                                                                                                                                                                                            two_gram_signals = await self.two_gram_detector.analyze_sequence(market_sequence, context=market_data)

                                                                                                                                                                                                                                                            # Process significant signals
                                                                                                                                                                                                                                                            significant_signals = [
                                                                                                                                                                                                                                                            s for s in two_gram_signals if s.burst_score > self.two_gram_detector.burst_threshold
                                                                                                                                                                                                                                                            ]

                                                                                                                                                                                                                                                                if significant_signals:
                                                                                                                                                                                                                                                                logger.info(f"🧬 Detected {len(significant_signals)} significant 2-gram patterns")

                                                                                                                                                                                                                                                                # Process signals through trading strategy executor
                                                                                                                                                                                                                                                                    for two_gram_signal in significant_signals:
                                                                                                                                                                                                                                                                    execution_result = await self.trading_strategy_executor.process_2gram_signal(
                                                                                                                                                                                                                                                                    two_gram_signal, market_data
                                                                                                                                                                                                                                                                    )
                                                                                                                                                                                                                                                                        if execution_result and execution_result.executed:
                                                                                                                                                                                                                                                                        logger.info(
                                                                                                                                                                                                                                                                        f"✅ Strategy executed: {execution_result.strategy_type.value} - "
                                                                                                                                                                                                                                                                        f"{execution_result.symbol}"
                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                        # Update performance metrics
                                                                                                                                                                                                                                                                        self.total_trades += 1

                                                                                                                                                                                                                                                                        # Log trade execution
                                                                                                                                                                                                                                                                        await self._log_trade_execution(two_gram_signal, execution_result, market_data)
                                                                                                                                                                                                                                                                            elif execution_result:
                                                                                                                                                                                                                                                                            logger.warning(f"⚠️ Strategy execution failed: {execution_result.error_message}")

                                                                                                                                                                                                                                                                            # Generate strategy triggers (legacy)
                                                                                                                                                                                                                                                                            triggers = await self.strategy_router.process_market_data(market_data)

                                                                                                                                                                                                                                                                            # Execute triggers
                                                                                                                                                                                                                                                                                for trigger in triggers:
                                                                                                                                                                                                                                                                                await self._execute_strategy_trigger(trigger, market_data)

                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                    logger.error(f"Error processing market data with 2-gram: {e}")

                                                                                                                                                                                                                                                                                        def _convert_market_data_to_sequence(self, market_data: Dict[str, Any]) -> str:
                                                                                                                                                                                                                                                                                        """Convert market data to character sequence for 2-gram analysis."""
                                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                                            sequence = ""

                                                                                                                                                                                                                                                                                            # Price direction signals
                                                                                                                                                                                                                                                                                                for asset, data in market_data.items():
                                                                                                                                                                                                                                                                                                    if isinstance(data, dict) and "price" in data:
                                                                                                                                                                                                                                                                                                    price_change = data.get("price_change_24h", 0)

                                                                                                                                                                                                                                                                                                        if price_change > 2.0:
                                                                                                                                                                                                                                                                                                        sequence += "U"  # Up
                                                                                                                                                                                                                                                                                                            elif price_change < -2.0:
                                                                                                                                                                                                                                                                                                            sequence += "D"  # Down
                                                                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                                                sequence += "C"  # Consolidation

                                                                                                                                                                                                                                                                                                                # Volume signals
                                                                                                                                                                                                                                                                                                                    for asset, data in market_data.items():
                                                                                                                                                                                                                                                                                                                        if isinstance(data, dict) and "volume" in data:
                                                                                                                                                                                                                                                                                                                        volume_change = data.get("volume_change_24h", 0)

                                                                                                                                                                                                                                                                                                                            if volume_change > 50.0:
                                                                                                                                                                                                                                                                                                                            sequence += "H"  # High volume
                                                                                                                                                                                                                                                                                                                                elif volume_change < -30.0:
                                                                                                                                                                                                                                                                                                                                sequence += "L"  # Low volume
                                                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                                    sequence += "N"  # Normal volume

                                                                                                                                                                                                                                                                                                                                    # Asset type signals
                                                                                                                                                                                                                                                                                                                                        if "BTC" in market_data:
                                                                                                                                                                                                                                                                                                                                        sequence += "B"
                                                                                                                                                                                                                                                                                                                                            if "ETH" in market_data:
                                                                                                                                                                                                                                                                                                                                            sequence += "E"
                                                                                                                                                                                                                                                                                                                                                if "USDC" in market_data:
                                                                                                                                                                                                                                                                                                                                                sequence += "S"  # Stable

                                                                                                                                                                                                                                                                                                                                            return sequence

                                                                                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                                                                                logger.error(f"Error converting market data to sequence: {e}")
                                                                                                                                                                                                                                                                                                                                            return ""

                                                                                                                                                                                                                                                                                                                                                async def _execute_strategy_trigger(self, trigger, market_data: Dict[str, Any]):
                                                                                                                                                                                                                                                                                                                                                """Execute a strategy trigger through the trading pipeline."""
                                                                                                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                                                                                                    logger.info(f"🎯 Executing strategy trigger: {trigger.strategy_name}")

                                                                                                                                                                                                                                                                                                                                                    # Execute through strategy router
                                                                                                                                                                                                                                                                                                                                                    execution_result = await self.strategy_router.execute_trigger(trigger)

                                                                                                                                                                                                                                                                                                                                                        if execution_result.execution_success:
                                                                                                                                                                                                                                                                                                                                                        logger.info(f"✅ Strategy execution successful: {trigger.strategy_name}")

                                                                                                                                                                                                                                                                                                                                                        # Update performance metrics
                                                                                                                                                                                                                                                                                                                                                        self.total_trades += 1
                                                                                                                                                                                                                                                                                                                                                            if hasattr(execution_result, 'pnl'):
                                                                                                                                                                                                                                                                                                                                                            self.total_profit += execution_result.pnl

                                                                                                                                                                                                                                                                                                                                                            # Log trade to registry
                                                                                                                                                                                                                                                                                                                                                            await self._log_trade_execution(trigger, execution_result, market_data)

                                                                                                                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                                                                                                logger.warning(f"⚠️ Strategy execution failed: {trigger.strategy_name}")

                                                                                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                    logger.error(f"Error executing strategy trigger: {e}")

                                                                                                                                                                                                                                                                                                                                                                        async def _strategy_processing_loop(self):
                                                                                                                                                                                                                                                                                                                                                                        """Strategy processing loop for continuous strategy evaluation."""
                                                                                                                                                                                                                                                                                                                                                                        logger.info("🎯 Starting strategy processing loop...")

                                                                                                                                                                                                                                                                                                                                                                            while self.running and not self.shutdown_requested:
                                                                                                                                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                                                                                                                                # Process any pending strategy triggers
                                                                                                                                                                                                                                                                                                                                                                                    if hasattr(self.strategy_router, 'pending_triggers'):
                                                                                                                                                                                                                                                                                                                                                                                        for trigger in self.strategy_router.pending_triggers:
                                                                                                                                                                                                                                                                                                                                                                                        await self._execute_strategy_trigger(trigger, {})

                                                                                                                                                                                                                                                                                                                                                                                        # Wait for next cycle
                                                                                                                                                                                                                                                                                                                                                                                        await asyncio.sleep(self.config.get("strategy_processing_interval", 5.0))

                                                                                                                                                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                            logger.error(f"Error in strategy processing loop: {e}")
                                                                                                                                                                                                                                                                                                                                                                                            await asyncio.sleep(5.0)

                                                                                                                                                                                                                                                                                                                                                                                                async def _performance_monitoring_loop(self):
                                                                                                                                                                                                                                                                                                                                                                                                """Performance monitoring and reporting loop."""
                                                                                                                                                                                                                                                                                                                                                                                                logger.info("📈 Starting performance monitoring loop...")

                                                                                                                                                                                                                                                                                                                                                                                                    while self.running and not self.shutdown_requested:
                                                                                                                                                                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                                                                                                                                                                        # Get performance metrics
                                                                                                                                                                                                                                                                                                                                                                                                        pipeline_summary = self.trading_pipeline.get_pipeline_summary()
                                                                                                                                                                                                                                                                                                                                                                                                        # Log performance metrics
                                                                                                                                                                                                                                                                                                                                                                                                        logger.info("📊 Performance Update:")
                                                                                                                                                                                                                                                                                                                                                                                                        logger.info(f"  Total Trades: {self.total_trades}")
                                                                                                                                                                                                                                                                                                                                                                                                        logger.info(f"  Total Profit: ${self.total_profit:.2f}")
                                                                                                                                                                                                                                                                                                                                                                                                        logger.info(f"  Active Positions: {len(self.current_positions)}")
                                                                                                                                                                                                                                                                                                                                                                                                        logger.info(f"  Pipeline Health: {pipeline_summary.get('status', 'unknown')}")
                                                                                                                                                                                                                                                                                                                                                                                                        # Check for system health issues
                                                                                                                                                                                                                                                                                                                                                                                                        health_check = await self.two_gram_detector.health_check()
                                                                                                                                                                                                                                                                                                                                                                                                            if health_check.get("overall_status") == "critical":
                                                                                                                                                                                                                                                                                                                                                                                                            logger.warning("🛡️ System health critical - activating T-cell protection")
                                                                                                                                                                                                                                                                                                                                                                                                            # Wait for next cycle
                                                                                                                                                                                                                                                                                                                                                                                                            await asyncio.sleep(self.config.get("performance_monitoring_interval", 30.0))
                                                                                                                                                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                logger.error(f"Error in performance monitoring loop: {e}")
                                                                                                                                                                                                                                                                                                                                                                                                                await asyncio.sleep(30.0)

                                                                                                                                                                                                                                                                                                                                                                                                                    async def _log_trade_execution(self, trigger, execution_result, market_data: Dict[str, Any]):
                                                                                                                                                                                                                                                                                                                                                                                                                    """Log trade execution to registry."""
                                                                                                                                                                                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                                                                                                                                                                                        trade_log = {
                                                                                                                                                                                                                                                                                                                                                                                                                        "timestamp": time.time(),
                                                                                                                                                                                                                                                                                                                                                                                                                        "trigger_type": trigger.trigger_type.value,
                                                                                                                                                                                                                                                                                                                                                                                                                        "strategy_name": trigger.strategy_name,
                                                                                                                                                                                                                                                                                                                                                                                                                        "pattern_data": trigger.pattern_data,
                                                                                                                                                                                                                                                                                                                                                                                                                        "execution_success": execution_result.execution_success,
                                                                                                                                                                                                                                                                                                                                                                                                                        "execution_time_ms": execution_result.execution_time_ms,
                                                                                                                                                                                                                                                                                                                                                                                                                        "pnl": getattr(execution_result, 'pnl', 0.0),
                                                                                                                                                                                                                                                                                                                                                                                                                        "market_data": market_data,
                                                                                                                                                                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                                                                                                                                                                        # Log to file or database
                                                                                                                                                                                                                                                                                                                                                                                                                        log_file = Path("logs/trade_executions.jsonl")
                                                                                                                                                                                                                                                                                                                                                                                                                        log_file.parent.mkdir(exist_ok=True)
                                                                                                                                                                                                                                                                                                                                                                                                                            with open(log_file, "a") as f:
                                                                                                                                                                                                                                                                                                                                                                                                                            f.write(json.dumps(trade_log) + "\n")
                                                                                                                                                                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                logger.error(f"Error logging trade execution: {e}")

                                                                                                                                                                                                                                                                                                                                                                                                                                    async def stop_trading(self):
                                                                                                                                                                                                                                                                                                                                                                                                                                    """Stop live trading operations."""
                                                                                                                                                                                                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                                                                                                                                                                                                        logger.info("🛑 Stopping live trading operations...")
                                                                                                                                                                                                                                                                                                                                                                                                                                        self.running = False
                                                                                                                                                                                                                                                                                                                                                                                                                                        self.shutdown_requested = True
                                                                                                                                                                                                                                                                                                                                                                                                                                        # Stop execution engine
                                                                                                                                                                                                                                                                                                                                                                                                                                            if self.execution_engine:
                                                                                                                                                                                                                                                                                                                                                                                                                                            await self.execution_engine.stop()
                                                                                                                                                                                                                                                                                                                                                                                                                                            # Stop all tasks
                                                                                                                                                                                                                                                                                                                                                                                                                                            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                                                                                                                                                                                                                                                                                                                                                                                                                                                for task in tasks:
                                                                                                                                                                                                                                                                                                                                                                                                                                                task.cancel()
                                                                                                                                                                                                                                                                                                                                                                                                                                                await asyncio.gather(*tasks, return_exceptions=True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                logger.info("✅ Live trading stopped successfully")
                                                                                                                                                                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                                    logger.error(f"Error stopping trading: {e}")

                                                                                                                                                                                                                                                                                                                                                                                                                                                        def get_system_status(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                                                                                                                                                                                                        """Get comprehensive system status."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                                                                                                                                                                                                            uptime = time.time() - self.start_time if self.start_time else 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                        return {
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "status": "running" if self.running else "stopped",
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "mode": self.mode,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "uptime_seconds": uptime,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "total_trades": self.total_trades,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "total_profit": self.total_profit,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "active_positions": len(self.current_positions),
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "portfolio": {
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "total_value": self.total_portfolio_value,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "held_assets": self.held_assets,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "available_pairs": len(self.active_pairs),
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "exchanges": list(self.available_balances.keys()),
                                                                                                                                                                                                                                                                                                                                                                                                                                                        },
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "components": {
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "two_gram_detector": self.two_gram_detector is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "strategy_router": self.strategy_router is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "trading_pipeline": self.trading_pipeline is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "execution_engine": self.execution_engine is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "entropy_executor": self.entropy_executor is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "portfolio_balancer": self.portfolio_balancer is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "btc_integration": self.btc_integration is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "master_system": self.master_system is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "ccxt_executor": self.ccxt_executor is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "order_book_manager": self.order_book_manager is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        "trading_strategy_executor": self.trading_strategy_executor is not None,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        },
                                                                                                                                                                                                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                                            logger.error(f"Error getting system status: {e}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                        return {"status": "error", "error": str(e)}


                                                                                                                                                                                                                                                                                                                                                                                                                                                            async def main():
                                                                                                                                                                                                                                                                                                                                                                                                                                                            """Main CLI entry point."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                            parser = argparse.ArgumentParser(description="Schwabot Trading System CLI")
                                                                                                                                                                                                                                                                                                                                                                                                                                                            parser.add_argument(
                                                                                                                                                                                                                                                                                                                                                                                                                                                            "--mode",
                                                                                                                                                                                                                                                                                                                                                                                                                                                            choices=["demo", "live", "backtest"],
                                                                                                                                                                                                                                                                                                                                                                                                                                                            default="demo",
                                                                                                                                                                                                                                                                                                                                                                                                                                                            help="Trading mode (default: demo)",
                                                                                                                                                                                                                                                                                                                                                                                                                                                            )
                                                                                                                                                                                                                                                                                                                                                                                                                                                            parser.add_argument("--config", default="config/schwabot_config.yaml", help="Configuration file path")
                                                                                                                                                                                                                                                                                                                                                                                                                                                            parser.add_argument("--log-level", default="INFO", help="Logging level")
                                                                                                                                                                                                                                                                                                                                                                                                                                                            args = parser.parse_args()
                                                                                                                                                                                                                                                                                                                                                                                                                                                            # Setup logging
                                                                                                                                                                                                                                                                                                                                                                                                                                                            setup_logging(level=getattr(logging, args.log_level.upper()))
                                                                                                                                                                                                                                                                                                                                                                                                                                                            # Create CLI instance
                                                                                                                                                                                                                                                                                                                                                                                                                                                            cli = SchwabotCLI(args.config)

                                                                                                                                                                                                                                                                                                                                                                                                                                                            # Setup signal handlers
                                                                                                                                                                                                                                                                                                                                                                                                                                                                def signal_handler(signum, frame):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                logger.info("Received shutdown signal")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                asyncio.create_task(cli.stop_trading())

                                                                                                                                                                                                                                                                                                                                                                                                                                                                signal.signal(signal.SIGINT, signal_handler)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                signal.signal(signal.SIGTERM, signal_handler)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # Initialize system
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        if not await cli.initialize_system(args.mode):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        logger.error("❌ System initialization failed")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        sys.exit(1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # Start live trading
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            if not await cli.start_live_trading():
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            logger.error("❌ Failed to start live trading")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            sys.exit(1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # Keep running until shutdown
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                while cli.running and not cli.shutdown_requested:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                await asyncio.sleep(1.0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # Stop trading
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                await cli.stop_trading()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # Print final status
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                final_status = cli.get_system_status()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                logger.info("📊 Final System Status:")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                logger.info(f"  Total Trades: {final_status['total_trades']}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                logger.info(f"  Total Profit: ${final_status['total_profit']:.2f}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                logger.info(f"  Uptime: {final_status['uptime_seconds']:.1f} seconds")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    logger.error(f"❌ Fatal error: {e}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    sys.exit(1)


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        asyncio.run(main())
