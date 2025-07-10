#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.advanced_settings_engine import AdvancedSettingsEngine
from core.api.cache_sync import CacheSyncService
from core.api.handlers.alt_fear_greed import FearGreedHandler
from core.api.handlers.coingecko import CoinGeckoHandler
from core.api.handlers.glassnode import GlassnodeHandler
from core.api.handlers.whale_alert import WhaleAlertHandler
from symbolic_profit_router import SymbolicProfitRouter

try:
    from core.risk_manager import RiskManager
    from core.strategy_logic import StrategyLogic
    from core.trade_executor import TradeExecutor
        BTC256SHAPipeline,
        FerrisWheelVisualizer,
        UnifiedMathematicsFramework,
        unified_trading_math,
    )
    except ImportError:
    # Fallback implementations
    StrategyLogic = None
    TradeExecutor = None
    RiskManager = None
    UnifiedMathematicsFramework = None
    BTC256SHAPipeline = None
    FerrisWheelVisualizer = None
    unified_trading_math = None


"""
Schwabot Enhanced Launcher
==========================

Comprehensive launcher that integrates:
- CacheSyncService with multiple API handlers
- Advanced Settings Engine with weighted confidence frameworks
- Unified Mathematics Framework
- Trading engine with cached sentiment/on-chain data integration
- Echo-based signal processing pipeline

This launcher implements the complete Schwabot spatial momentum system
with recursive logic and adaptive memory.
"""

logger = logging.getLogger(__name__)


class EnhancedDataIntegrator:
    """Integrates cached API data into trading signals and strategy logic."""

    def __init__()
        self,
        settings_engine: AdvancedSettingsEngine,
        math_framework: UnifiedMathematicsFramework,
    ):
        self.settings_engine = settings_engine
        self.math_framework = math_framework
        self.data_cache: Dict[str, Any] = {}
        self.signal_weights: Dict[str, float] = {}
            "fear_greed": 0.3,
            "whale_activity": 0.4,
            "onchain_metrics": 0.5,
            "market_sentiment": 0.3,
        }

    async def integrate_cached_data(self) -> Dict[str, float]:
        """Integrate all cached data sources into unified trading signals."""
        signals = {}

        try:
            # Fear & Greed Index
            fear_greed_data = await self._load_cached_data("sentiment/latest.json")
            if fear_greed_data:
                fg_signal = self._process_fear_greed_signal(fear_greed_data)
                signals["fear_greed"] = fg_signal

            # Whale Activity
            whale_data = await self._load_cached_data("whale_data/latest.json")
            if whale_data:
                whale_signal = self._process_whale_signal(whale_data)
                signals["whale_activity"] = whale_signal

            # On-chain Metrics
            onchain_data = await self._load_cached_data("onchain_data/latest.json")
            if onchain_data:
                onchain_signal = self._process_onchain_signal(onchain_data)
                signals["onchain_metrics"] = onchain_signal

            # Market Data
            market_data = await self._load_cached_data("market_data/latest.json")
            if market_data:
                market_signal = self._process_market_signal(market_data)
                signals["market_sentiment"] = market_signal

            # Apply settings bias to signals
            biased_signals = {}
            for signal_name, signal_value in signals.items():
                biased_value = self.settings_engine.apply_bias_to_module()
                    f"signal_{signal_name}", signal_value
                )
                biased_signals[signal_name] = biased_value

            return biased_signals

        except Exception as e:
            logger.error(f"Failed to integrate cached data: {e}")
            return {}

    async def _load_cached_data(self, cache_path: str) -> Optional[Dict]:
        """Load data from cache file."""
        try:
            cache_file = Path("flask/feeds") / cache_path
            if cache_file.exists():

                with open(cache_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cached data from {cache_path}: {e}")
        return None

    def _process_fear_greed_signal(self, data: Dict) -> float:
        """Process Fear & Greed Index into trading signal (-1 to 1)."""
        try:
            fg_value = data.get("value", 50)

            # Convert to signal: extreme fear = buy signal, extreme greed = sell signal
            if fg_value <= 20:  # Extreme fear
                return 0.8  # Strong buy signal
            elif fg_value <= 40:  # Fear
                return 0.4  # Moderate buy signal
            elif fg_value <= 60:  # Neutral
                return 0.0  # No signal
            elif fg_value <= 80:  # Greed
                return -0.4  # Moderate sell signal
            else:  # Extreme greed
                return -0.8  # Strong sell signal

        except Exception as e:
            logger.error(f"Failed to process fear greed signal: {e}")
            return 0.0

    def _process_whale_signal(self, data: Dict) -> float:
        """Process whale transaction data into trading signal."""
        try:
            summary = data.get("summary", {})
            whale_score = summary.get("whale_activity_score", 0)
            summary.get("total_volume_usd", 0)

            # High whale activity suggests institutional movement
            if whale_score > 70:
                return 0.6  # Follow whale movement
            elif whale_score > 40:
                return 0.3  # Moderate signal
            else:
                return 0.0  # No significant whale activity

        except Exception as e:
            logger.error(f"Failed to process whale signal: {e}")
            return 0.0

    def _process_onchain_signal(self, data: Dict) -> float:
        """Process on-chain metrics into trading signal."""
        try:
            scores = data.get("composite_scores", {})
            network_health = scores.get("network_health", 50)
            valuation = scores.get("valuation", 50)
            momentum = scores.get("momentum", 50)

            # Weighted combination of scores
            combined_score = network_health * 0.3 + valuation * 0.4 + momentum * 0.3

            # Convert to signal (-1 to 1)
            normalized_signal = (combined_score - 50) / 50
            return max(-1.0, min(1.0, normalized_signal))

        except Exception as e:
            logger.error(f"Failed to process onchain signal: {e}")
            return 0.0

    def _process_market_signal(self, data: Dict) -> float:
        """Process market sentiment into trading signal."""
        try:
            sentiment = data.get("market_sentiment", {})
            bullish_ratio = sentiment.get("bullish_ratio", 0.5)
            market_trend = sentiment.get("market_trend", "neutral")

            # Convert trend to numeric
            trend_values = {}
                "very_bearish": -1.0,
                "bearish": -0.5,
                "neutral": 0.0,
                "bullish": 0.5,
                "very_bullish": 1.0,
            }

            trend_signal = trend_values.get(market_trend, 0.0)
            bullish_signal = (bullish_ratio - 0.5) * 2  # Convert to -1 to 1

            # Combine signals
            combined_signal = (trend_signal * 0.6) + (bullish_signal * 0.4)
            return max(-1.0, min(1.0, combined_signal))

        except Exception as e:
            logger.error(f"Failed to process market signal: {e}")
            return 0.0


class SchawbotEnhancedLauncher:
    """
    Enhanced launcher implementing Schwabot's complete spatial momentum system'
    with recursive logic, adaptive memory, and multi-source data integration.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("settings/launcher_config.json")
        self.running = False
        self.tasks: Dict[str, asyncio.Task] = {}

        # Core components
        self.math_framework: Optional[UnifiedMathematicsFramework] = None
        self.settings_engine: Optional[AdvancedSettingsEngine] = None
        self.cache_sync_service: Optional[CacheSyncService] = None
        self.data_integrator: Optional[EnhancedDataIntegrator] = None
        self.profit_router: Optional[SymbolicProfitRouter] = None

        # Trading components
        self.strategy_logic: Optional[StrategyLogic] = None
        self.trade_executor: Optional[TradeExecutor] = None
        self.risk_manager: Optional[RiskManager] = None

        # Visualization and monitoring
        self.btc_pipeline: Optional[BTC256SHAPipeline] = None
        self.ferris_visualizer: Optional[FerrisWheelVisualizer] = None

        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.start_time = 0.0

    async def initialize(self) -> bool:
        """Initialize all Schwabot components."""
        try:
            logger.info("🚀 Initializing Schwabot Enhanced System...")
            self.start_time = time.time()

            # Load settings engine
            self.settings_engine = AdvancedSettingsEngine()
            await self.settings_engine.load_configuration(self.config_path)

            # Initialize math framework
            self.math_framework = UnifiedMathematicsFramework()

            # Initialize cache sync service
            self.cache_sync_service = CacheSyncService()
            await self._register_api_handlers()

            # Initialize data integrator
            self.data_integrator = EnhancedDataIntegrator()
                self.settings_engine, self.math_framework
            )

            # Initialize trading components
            await self._initialize_trading_components()

            # Initialize symbolic profit router
            self.profit_router = SymbolicProfitRouter()

            # Initialize BTC pipeline and visualizer
            if BTC256SHAPipeline and FerrisWheelVisualizer:
                self.btc_pipeline = BTC256SHAPipeline()
                self.ferris_visualizer = FerrisWheelVisualizer()

            logger.info("✅ Schwabot Enhanced System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Schwabot: {e}")
            return False

    async def _register_api_handlers(self) -> None:
        """Register all API handlers with the cache sync service."""
        # Fear & Greed Index handler
        self.cache_sync_service.register_handler()
            handler_name="fear_greed",
            handler=FearGreedHandler(),
            update_interval=self.settings_engine.get_setting_value()
                "fear_greed_update_interval", 3600
            ),
        )

        # Whale Alert handler
        self.cache_sync_service.register_handler()
            handler_name="whale_alerts",
            handler=WhaleAlertHandler(),
            update_interval=self.settings_engine.get_setting_value()
                "whale_alert_update_interval", 600
            ),
        )

        # Glassnode handler
        self.cache_sync_service.register_handler()
            handler_name="glassnode",
            handler=GlassnodeHandler()
                api_key=self.settings_engine.get_api_key("glassnode")
            ),
            update_interval=self.settings_engine.get_setting_value()
                "glassnode_update_interval", 7200
            ),
        )

        # CoinGecko handler
        self.cache_sync_service.register_handler()
            handler_name="coingecko",
            handler=CoinGeckoHandler(),
            update_interval=self.settings_engine.get_setting_value()
                "coingecko_update_interval", 300
            ),
        )
        logger.info(f"Registered {len(self.cache_sync_service.handlers)} API handlers")

    async def _initialize_trading_components(self) -> None:
        """Initialize trading components if available."""
        try:
            if StrategyLogic and TradeExecutor and RiskManager:
                self.strategy_logic = StrategyLogic(self.settings_engine)
                self.trade_executor = TradeExecutor(self.settings_engine)
                self.risk_manager = RiskManager(self.settings_engine)
                logger.info("Trading components initialized.")
            else:
                logger.warning("Trading components not available, running in data-only mode.")
        except Exception as e:
            logger.warning(f"Some trading components not available: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start(self) -> None:
        """Start all Schwabot services and the main trading loop."""
        if not await self.initialize():
            logger.error("Failed to initialize, cannot start")
            return

        self.running = True
        self._setup_signal_handlers()

        logger.info("🚀 Starting Schwabot services...")

        # Start cache sync service
        self.tasks["cache_sync"] = asyncio.create_task()
            self.cache_sync_service.start()
        )

        # Start monitoring tasks
        self.tasks["monitor_cache"] = asyncio.create_task(self._monitor_cache_sync())
        self.tasks["monitor_settings"] = asyncio.create_task(self._monitor_settings())

        # Start data integration loop
        self.tasks["data_integration"] = asyncio.create_task()
            self._data_integration_loop()
        )

        # Start main trading loop
        self.tasks["trading_loop"] = asyncio.create_task(self._enhanced_trading_loop())

        # Start performance monitor
        self.tasks["performance_monitor"] = asyncio.create_task()
            self._performance_monitor()
        )

        # Keep running until shutdown signal
        while self.running:
            await asyncio.sleep(1)

        logger.info("Schwabot main loop finished.")
        # Final shutdown sequence
        await self.shutdown()

    async def _monitor_cache_sync(self) -> None:
        """Monitor cache sync service health."""
        while self.running:
            try:
                last_update_times = self.cache_sync_service.get_last_update_times()
                # Log or check update times
                logger.debug(f"Cache last updated: {last_update_times}")
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error monitoring cache sync: {e}")
                await asyncio.sleep(30)

    async def _monitor_settings(self) -> None:
        """Monitor settings engine and apply adaptive recommendations."""
        while self.running:
            try:
                # Check for settings drift
                drift = self.settings_engine.check_drift()
                if drift:
                    logger.info(f"Settings drift detected: {drift}")
                    # Apply adaptive changes
                    self.settings_engine.apply_recommendations(drift)

                # Reload settings if file changed
                if await self.settings_engine.check_for_updates():
                    logger.info("Configuration file changed, reloading settings...")
                    await self.settings_engine.load_configuration(self.config_path)

                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in settings monitor: {e}")
                await asyncio.sleep(60)

    async def _data_integration_loop(self) -> None:
        """Main data integration loop combining all signals."""
        while self.running:
            try:
                # Integrate all data sources to get unified signals
                signals = await self.data_integrator.integrate_cached_data()

                # Calculate unified signal score
                unified_signal = sum(signals.values()) / len(signals) if signals else 0.0

                # Update performance metrics
                self.performance_metrics["last_unified_signal"] = unified_signal
                self.performance_metrics["signal_breakdown"] = signals

                logger.debug(f"Unified Signal: {unified_signal:.3f}, Breakdown: {signals}")

                await asyncio.sleep(self.settings_engine.get_setting_value())
                    "data_integration_interval", 60
                ))
            except Exception as e:
                logger.error(f"Error in data integration loop: {e}")
                await asyncio.sleep(60)

    async def _enhanced_trading_loop(self) -> None:
        """Enhanced trading loop with spatial momentum system."""
        while self.running:
            try:
                # Get current market data
                await self._process_market_tick()

                # Get unified signals
                unified_signal = self.performance_metrics.get()
                    "last_unified_signal", 0.0
                )
                signal_breakdown = self.performance_metrics.get("signal_breakdown", {})

                # Apply spatial momentum calculations
                await self._apply_spatial_momentum(unified_signal, signal_breakdown)

                # Execute trades if signal is strong enough
                await self._execute_trading_decisions(unified_signal)

                # Update performance metrics
                await self._update_trading_performance()

                await asyncio.sleep(10)  # Main trading loop frequency

            except Exception as e:
                logger.error(f"Error in enhanced trading loop: {e}")
                await asyncio.sleep(30)

    async def _process_market_tick(self) -> None:
        """Process current market tick data."""
        try:
            # Simulate market data processing
            current_price = 50000.0  # This would come from real market data
            current_volume = 1000.0
            timestamp = time.time()

            # Process through BTC pipeline
            if self.btc_pipeline:
                pipeline_result = self.btc_pipeline.process_price_data()
                    current_price, timestamp
                )

                # Update performance metrics
                self.performance_metrics["current_price"] = current_price
                self.performance_metrics["current_volume"] = current_volume
                self.performance_metrics["pipeline_result"] = pipeline_result

        except Exception as e:
            logger.error(f"Error processing market tick: {e}")

    async def _apply_spatial_momentum(self, unified_signal, breakdown) -> None:
        """Apply spatial momentum calculations to trading decisions."""
        try:
            # Calculate momentum vector using unified math framework
            if self.math_framework:
                drift_field = self.math_framework.compute_unified_drift_field()
                    x=unified_signal,
                    y=time.time() / 1000,
                    z=0.5,
                    time=time.time() / 86400,
                )

                # Store momentum calculations
                self.performance_metrics["drift_field"] = drift_field
                self.performance_metrics["momentum_calculation"] = {}
                    "unified_signal": unified_signal,
                    "signal_breakdown": breakdown,
                    "drift_field": drift_field,
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"Error in spatial momentum calculation: {e}")

    async def _execute_trading_decisions(self, unified_signal) -> None:
        """Execute trading decisions based on unified signal."""
        try:
            # Apply signal threshold from settings
            threshold = ()
                self.settings_engine.get_setting_value("ghost_relay_threshold") or 0.9
            )

            if abs(unified_signal) >= threshold:
                # Strong signal - consider trading
                if self.profit_router:
                    # Route through symbolic profit router
                    profit_signal = await self._calculate_profit_optimization()
                        unified_signal
                    )

                    if profit_signal > 0.1:  # Minimum profit threshold
                        logger.info()
                            f"💰 Profit opportunity detected: {profit_signal:.3f}"
                        )

                        # Update settings feedback
                        self.settings_engine.update_profit_feedback()
                            "unified_trading", profit_signal
                        )

        except Exception as e:
            logger.error(f"Error in trading execution: {e}")

    async def _calculate_profit_optimization(self, signal) -> float:
        """Calculate profit optimization using brain glyph processing."""
        try:
            # Use unified trading math for profit calculation
            current_price = self.performance_metrics.get("current_price", 50000.0)
            current_volume = self.performance_metrics.get("current_volume", 1000.0)

            # Apply brain glyph processing
            profit_score = unified_trading_math.calculate_profit_optimization()
                current_price, current_volume, "BTC"
            )

            # Apply signal weighting
            weighted_profit = profit_score * abs(signal)

            return weighted_profit

        except Exception as e:
            logger.error(f"Error calculating profit optimization: {e}")
            return 0.0

    async def _update_trading_performance(self) -> None:
        """Update trading performance metrics."""
        try:
            current_time = time.time()
            runtime = current_time - self.start_time

            self.performance_metrics.update()
                {}
                    "runtime_seconds": runtime,
                    "runtime_hours": runtime / 3600,
                    "system_status": self.settings_engine.get_system_status(),
                    "math_framework_status": self.math_framework.get_system_status(),
                    "last_update": current_time,
                }
            )

        except Exception as e:
            logger.error(f"Error updating performance: {e}")

    async def _performance_monitor(self) -> None:
        """Monitor system performance and log key metrics."""
        while self.running:
            try:
                runtime = time.time() - self.start_time

                # Log performance every 15 minutes
                if runtime % 900 < 30:  # Within 30 seconds of 15-minute mark
                    logger.info()
                        f"📊 Performance Report (Runtime: {runtime / 3600:.1f}h)"
                    )
                    logger.info()
                        f"   Unified Signal: {"}
                            self.performance_metrics.get()
                                'last_unified_signal',
                                0):.3f}")"
                    logger.info()
                        f"   Drift Field: {self.performance_metrics.get('drift_field', 0):.3f}"
                    )
                    logger.info()
                        f"   Active Settings: {len(self.settings_engine.settings_state)}"
                    )

                    # Save configuration periodically
                    self.settings_engine.save_configuration()

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)

    async def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        logger.info("🛑 Initiating Schwabot Enhanced System shutdown...")
        self.running = False

        # Cancel all tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                logger.info(f"Cancelling {task_name} task...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop cache sync service
        if self.cache_sync_service:
            await self.cache_sync_service.stop()

        # Save final configuration
        if self.settings_engine:
            self.settings_engine.save_configuration()

        # Export final visualization data
        if self.ferris_visualizer:
            try:
                self.ferris_visualizer.create_visualization_data()
                    {"performance": self.performance_metrics}
                )
                self.ferris_visualizer.export_visualization_data()
                    "final_session_data.json"
                )
            except Exception as e:
                logger.error(f"Failed to export visualization data: {e}")

        logger.info("✅ Schwabot Enhanced System shutdown complete")


async def main():
    """Main entry point for the enhanced launcher."""
    launcher = SchawbotEnhancedLauncher()
    await launcher.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Launcher terminated by user.")
    except Exception as e:
        logger.error(f"Unhandled exception in launcher: {e}")
        sys.exit(1)
