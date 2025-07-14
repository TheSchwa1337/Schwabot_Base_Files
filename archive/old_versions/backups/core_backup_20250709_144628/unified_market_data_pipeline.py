"""Module for Schwabot trading system."""


import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .api.handlers.alt_fear_greed import FearGreedHandler
from .api.handlers.coingecko import CoinGeckoHandler
from .api.handlers.glassnode import GlassnodeHandler
from .api.handlers.whale_alert import WhaleAlertHandler
from .clean_math_foundation import CleanMathFoundation
from .soulprint_registry import SoulprintRegistry

# -*- coding: utf-8 -*-
"""
Unified Market Data Pipeline for Schwabot Trading System

This pipeline integrates multiple API sources (CoinGecko, Glassnode, CoinMarketCap)
to provide clean, standardized, and optimized market data input for the trading system.

    Key Features:
    - Multi-source data aggregation and validation
    - Technical indicator calculations (RSI, MACD, Bollinger, Bands)
    - Sentiment analysis and on-chain metrics integration
    - Standardized MarketDataPacket output
    - Real-time data cleaning and optimization
    - Caching and failover mechanisms
    - Registry-compatible data structure

    """

    # Import API handlers
    # Import math utilities
    logger = logging.getLogger(__name__)


        class DataSource(Enum):
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Supported data sources for market data."""

        COINGECKO = "coingecko"
        GLASSNODE = "glassnode"
        COINMARKETCAP = "coinmarketcap"
        FEAR_GREED = "fear_greed"
        WHALE_ALERT = "whale_alert"


            class DataQuality(Enum):
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Data quality levels."""

            EXCELLENT = "excellent"  # All sources available, fresh data
            GOOD = "good"  # Most sources available, recent data
            ACCEPTABLE = "acceptable"  # Some sources available, older data
            POOR = "poor"  # Limited sources, stale data
            FAILED = "failed"  # No reliable data available


            @dataclass
                class TechnicalIndicators:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Technical indicators calculated from price data."""

                rsi_14: float = 50.0
                rsi_21: float = 50.0
                macd_line: float = 0.0
                macd_signal: float = 0.0
                macd_histogram: float = 0.0
                bb_upper: float = 0.0
                bb_middle: float = 0.0
                bb_lower: float = 0.0
                bb_position: float = 0.5
                sma_20: float = 0.0
                ema_12: float = 0.0
                ema_26: float = 0.0
                volume_sma_20: float = 0.0
                volume_ratio: float = 1.0
                atr_14: float = 0.0
                stoch_k: float = 50.0
                stoch_d: float = 50.0


                @dataclass
                    class OnChainMetrics:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """On-chain metrics from Glassnode and similar sources."""

                    hash_rate: float = 0.0
                    difficulty: float = 0.0
                    active_addresses: int = 0
                    transaction_count: int = 0
                    mvrv_ratio: float = 1.0
                    nvt_ratio: float = 50.0
                    sopr: float = 1.0
                    exchange_inflow: float = 0.0
                    exchange_outflow: float = 0.0
                    whale_activity_score: float = 0.0
                    network_health_score: float = 50.0
                    hodl_waves: Dict[str, float] = field(default_factory=dict)


                    @dataclass
                        class MarketSentiment:
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """Market sentiment indicators."""

                        fear_greed_index: float = 50.0
                        fear_greed_classification: str = "Neutral"
                        social_volume: float = 0.0
                        social_sentiment: float = 0.0
                        reddit_sentiment: float = 0.0
                        twitter_sentiment: float = 0.0
                        news_sentiment: float = 0.0
                        trending_score: float = 0.0
                        market_dominance: float = 0.0


                        @dataclass
                            class MarketDataPacket:
    """Class for Schwabot trading functionality."""
                            """Class for Schwabot trading functionality."""
                            """Standardized market data packet for trading pipeline input."""

                            # Basic price data
                            symbol: str
                            price: float
                            volume_24h: float
                            market_cap: float
                            price_change_24h: float
                            volume_change_24h: float

                            # Time and quality
                            timestamp: float
                            data_quality: DataQuality
                            source_count: int
                            freshness_score: float

                            # Technical indicators
                            technical_indicators: TechnicalIndicators

                            # On-chain metrics
                            onchain_metrics: OnChainMetrics

                            # Sentiment data
                            market_sentiment: MarketSentiment

                            # Derived metrics for trading system
                            volatility: float = 0.0
                            trend_strength: float = 0.0
                            entropy_level: float = 4.0
                            liquidity_score: float = 0.0
                            momentum_score: float = 0.0

                            # API source tracking
                            sources_used: List[str] = field(default_factory=list)
                            api_latencies: Dict[str, float] = field(default_factory=dict)

                            # Registry-compatible metadata
                            metadata: Dict[str, Any] = field(default_factory=dict)


                            @dataclass
                                class PipelineMetrics:
    """Class for Schwabot trading functionality."""
                                """Class for Schwabot trading functionality."""
                                """Pipeline performance and health metrics."""

                                total_requests: int = 0
                                successful_requests: int = 0
                                failed_requests: int = 0
                                average_latency: float = 0.0
                                cache_hit_rate: float = 0.0
                                data_quality_score: float = 0.0
                                uptime_percentage: float = 100.0
                                last_update: float = 0.0


                                    class UnifiedMarketDataPipeline:
    """Class for Schwabot trading functionality."""
                                    """Class for Schwabot trading functionality."""
                                    """
                                    Unified Market Data Pipeline that aggregates, cleans, and optimizes
                                    market data from multiple sources for consistent trading system input.
                                    """

                                        def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
                                        """Initialize the unified market data pipeline."""
                                        self.config = config or self._default_config()

                                        # Mathematical foundation for calculations
                                        self.math_foundation = CleanMathFoundation()

                                        # API handlers initialization
                                        self.handlers: Dict[str, Any] = {}
                                        self._initialize_handlers()

                                        # Data caching and storage
                                        self.data_cache: Dict[str, MarketDataPacket] = {}
                                        self.cache_expiry: Dict[str, float] = {}
                                        self.price_history: Dict[str, List[float]] = {}
                                        self.volume_history: Dict[str, List[float]] = {}

                                        # Pipeline metrics
                                        self.metrics = PipelineMetrics()
                                        self.start_time = time.time()

                                        # Registry integration
                                        self.registry = None
                                            if self.config.get("registry_file"):
                                            self.registry = SoulprintRegistry(self.config["registry_file"])

                                            logger.info("Unified Market Data Pipeline initialized")

                                                def _default_config(self) -> Dict[str, Any]:
                                                """Default configuration for the pipeline."""
                                            return {
                                            "cache_ttl": 300,  # 5 minutes
                                            "max_price_history": 1000,
                                            "api_timeout": 30,
                                            "retry_attempts": 3,
                                            "failover_enabled": True,
                                            "quality_threshold": 0.7,
                                            "registry_file": None,
                                            "apis": {
                                            "coingecko": {"enabled": True, "weight": 0.4},
                                            "glassnode": {"enabled": True, "weight": 0.3},
                                            "fear_greed": {"enabled": True, "weight": 0.2},
                                            "whale_alert": {"enabled": True, "weight": 0.1},
                                            },
                                            }

                                                def _initialize_handlers(self) -> None:
                                                """Initialize API handlers based on configuration."""
                                                    try:
                                                        if self.config["apis"]["coingecko"]["enabled"]:
                                                        self.handlers["coingecko"] = CoinGeckoHandler()

                                                            if self.config["apis"]["glassnode"]["enabled"]:
                                                            api_key = self.config.get("glassnode_api_key")
                                                                if api_key:
                                                                self.handlers["glassnode"] = GlassnodeHandler(api_key=api_key)
                                                                    else:
                                                                    logger.warning("Glassnode API key not provided, handler disabled")

                                                                        if self.config["apis"]["fear_greed"]["enabled"]:
                                                                        self.handlers["fear_greed"] = FearGreedHandler()

                                                                            if self.config["apis"]["whale_alert"]["enabled"]:
                                                                            api_key = self.config.get("whale_alert_api_key")
                                                                                if api_key:
                                                                                self.handlers["whale_alert"] = WhaleAlertHandler(api_key=api_key)
                                                                                    else:
                                                                                    logger.warning("Whale Alert API key not provided, handler disabled")

                                                                                    logger.info("Initialized {0} API handlers".format(len(self.handlers)))

                                                                                        except Exception as e:
                                                                                        logger.error("Error initializing API handlers: {0}".format(e))

                                                                                            async def get_market_data(self, symbol: str, force_refresh: bool = False) -> MarketDataPacket:
                                                                                            """
                                                                                            Get unified market data for a symbol.

                                                                                                Args:
                                                                                                symbol: Trading symbol (e.g., "BTC", "ETH")
                                                                                                force_refresh: Force refresh from APIs

                                                                                                    Returns:
                                                                                                    Standardized MarketDataPacket
                                                                                                    """
                                                                                                    start_time = time.time()

                                                                                                        try:
                                                                                                        # Check cache first
                                                                                                            if not force_refresh and self._is_cached(symbol):
                                                                                                            cached_data = self.data_cache[symbol]
                                                                                                            logger.debug("Using cached data for {0}".format(symbol))
                                                                                                        return cached_data

                                                                                                        # Fetch data from all available sources
                                                                                                        raw_data = await self._fetch_from_all_sources(symbol)

                                                                                                        # Clean and validate data
                                                                                                        cleaned_data = self._clean_and_validate(raw_data, symbol)

                                                                                                        # Calculate technical indicators
                                                                                                        technical_indicators = self._calculate_technical_indicators(symbol, cleaned_data)

                                                                                                        # Extract on-chain metrics
                                                                                                        onchain_metrics = self._extract_onchain_metrics(cleaned_data)

                                                                                                        # Calculate sentiment
                                                                                                        market_sentiment = self._calculate_market_sentiment(cleaned_data)

                                                                                                        # Calculate derived metrics
                                                                                                        derived_metrics = self._calculate_derived_metrics(cleaned_data, technical_indicators)

                                                                                                        # Assess data quality
                                                                                                        data_quality, freshness_score = self._assess_data_quality(raw_data)

                                                                                                        # Create unified market data packet
                                                                                                        market_packet = MarketDataPacket(
                                                                                                        symbol=symbol,
                                                                                                        price=cleaned_data.get("price", 0.0),
                                                                                                        volume_24h=cleaned_data.get("volume_24h", 0.0),
                                                                                                        market_cap=cleaned_data.get("market_cap", 0.0),
                                                                                                        price_change_24h=cleaned_data.get("price_change_24h", 0.0),
                                                                                                        volume_change_24h=cleaned_data.get("volume_change_24h", 0.0),
                                                                                                        timestamp=time.time(),
                                                                                                        data_quality=data_quality,
                                                                                                        source_count=len([h for h in self.handlers.keys() if h in raw_data]),
                                                                                                        freshness_score=freshness_score,
                                                                                                        technical_indicators=technical_indicators,
                                                                                                        onchain_metrics=onchain_metrics,
                                                                                                        market_sentiment=market_sentiment,
                                                                                                        volatility=derived_metrics["volatility"],
                                                                                                        trend_strength=derived_metrics["trend_strength"],
                                                                                                        entropy_level=derived_metrics["entropy_level"],
                                                                                                        liquidity_score=derived_metrics["liquidity_score"],
                                                                                                        momentum_score=derived_metrics["momentum_score"],
                                                                                                        sources_used=list(raw_data.keys()),
                                                                                                        api_latencies={k: v.get("latency", 0.0) for k, v in raw_data.items()},
                                                                                                        metadata={
                                                                                                        "pipeline_version": "1.0.0",
                                                                                                        "calculation_time": time.time() - start_time,
                                                                                                        "quality_score": self._calculate_quality_score(data_quality, freshness_score),
                                                                                                        "cache_key": "{0}_{1}".format(symbol, int(time.time() // self.config['cache_ttl'])),
                                                                                                        },
                                                                                                        )

                                                                                                        # Cache the result
                                                                                                        self._cache_data(symbol, market_packet)

                                                                                                        # Update pipeline metrics
                                                                                                        self._update_metrics(True, time.time() - start_time)

                                                                                                        # Log to registry if enabled
                                                                                                            if self.registry:
                                                                                                            self._log_to_registry(market_packet)

                                                                                                            logger.info(
                                                                                                            "Generated market data packet for {0} "
                                                                                                            "(quality: {1}, sources: {2})".format(symbol, data_quality.value, len(raw_data))
                                                                                                            )

                                                                                                        return market_packet

                                                                                                            except Exception as e:
                                                                                                            logger.error("Error generating market data for {0}: {1}".format(symbol, e))
                                                                                                            self._update_metrics(False, time.time() - start_time)

                                                                                                            # Return fallback data
                                                                                                        return self._create_fallback_packet(symbol)

                                                                                                            async def _fetch_from_all_sources(self, symbol: str) -> Dict[str, Any]:
                                                                                                            """Fetch data from all available API sources."""
                                                                                                            raw_data = {}

                                                                                                            # Fetch from CoinGecko
                                                                                                                if "coingecko" in self.handlers:
                                                                                                                    try:
                                                                                                                    start_time = time.time()
                                                                                                                    cg_data = await self.handlers["coingecko"].get_data(force_refresh=True)
                                                                                                                    latency = time.time() - start_time

                                                                                                                        if cg_data and "coin_prices" in cg_data:
                                                                                                                        raw_data["coingecko"] = {
                                                                                                                        "data": cg_data,
                                                                                                                        "latency": latency,
                                                                                                                        "timestamp": time.time(),
                                                                                                                        }
                                                                                                                            except Exception as e:
                                                                                                                            logger.error("CoinGecko fetch failed: {0}".format(e))

                                                                                                                            # Fetch from Glassnode
                                                                                                                                if "glassnode" in self.handlers:
                                                                                                                                    try:
                                                                                                                                    start_time = time.time()
                                                                                                                                    gn_data = await self.handlers["glassnode"].get_data(force_refresh=True)
                                                                                                                                    latency = time.time() - start_time

                                                                                                                                        if gn_data and "latest_values" in gn_data:
                                                                                                                                        raw_data["glassnode"] = {
                                                                                                                                        "data": gn_data,
                                                                                                                                        "latency": latency,
                                                                                                                                        "timestamp": time.time(),
                                                                                                                                        }
                                                                                                                                            except Exception as e:
                                                                                                                                            logger.error("Glassnode fetch failed: {0}".format(e))

                                                                                                                                            # Fetch from Fear & Greed
                                                                                                                                                if "fear_greed" in self.handlers:
                                                                                                                                                    try:
                                                                                                                                                    start_time = time.time()
                                                                                                                                                    fg_data = await self.handlers["fear_greed"].get_data(force_refresh=True)
                                                                                                                                                    latency = time.time() - start_time

                                                                                                                                                        if fg_data:
                                                                                                                                                        raw_data["fear_greed"] = {
                                                                                                                                                        "data": fg_data,
                                                                                                                                                        "latency": latency,
                                                                                                                                                        "timestamp": time.time(),
                                                                                                                                                        }
                                                                                                                                                            except Exception as e:
                                                                                                                                                            logger.error("Fear & Greed fetch failed: {0}".format(e))

                                                                                                                                                            # Fetch from Whale Alert
                                                                                                                                                                if "whale_alert" in self.handlers:
                                                                                                                                                                    try:
                                                                                                                                                                    start_time = time.time()
                                                                                                                                                                    wa_data = await self.handlers["whale_alert"].get_data(force_refresh=True)
                                                                                                                                                                    latency = time.time() - start_time

                                                                                                                                                                        if wa_data:
                                                                                                                                                                        raw_data["whale_alert"] = {
                                                                                                                                                                        "data": wa_data,
                                                                                                                                                                        "latency": latency,
                                                                                                                                                                        "timestamp": time.time(),
                                                                                                                                                                        }
                                                                                                                                                                            except Exception as e:
                                                                                                                                                                            logger.error("Whale Alert fetch failed: {0}".format(e))

                                                                                                                                                                        return raw_data

                                                                                                                                                                            def _clean_and_validate(self, raw_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
                                                                                                                                                                            """Clean and validate raw API data."""
                                                                                                                                                                            cleaned = {
                                                                                                                                                                            "price": 0.0,
                                                                                                                                                                            "volume_24h": 0.0,
                                                                                                                                                                            "market_cap": 0.0,
                                                                                                                                                                            "price_change_24h": 0.0,
                                                                                                                                                                            "volume_change_24h": 0.0,
                                                                                                                                                                            }

                                                                                                                                                                            # Extract price data from CoinGecko
                                                                                                                                                                                if "coingecko" in raw_data:
                                                                                                                                                                                cg_data = raw_data["coingecko"]["data"]
                                                                                                                                                                                    if "coin_prices" in cg_data:
                                                                                                                                                                                    # Find symbol data
                                                                                                                                                                                        for coin_id, coin_data in cg_data["coin_prices"].items():
                                                                                                                                                                                            if symbol.lower() in coin_id.lower() or coin_id.lower() in symbol.lower():
                                                                                                                                                                                                if "usd" in coin_data:
                                                                                                                                                                                                cleaned["price"] = float(coin_data["usd"])
                                                                                                                                                                                                    if "usd_24h_vol" in coin_data:
                                                                                                                                                                                                    cleaned["volume_24h"] = float(coin_data["usd_24h_vol"])
                                                                                                                                                                                                        if "usd_market_cap" in coin_data:
                                                                                                                                                                                                        cleaned["market_cap"] = float(coin_data["usd_market_cap"])
                                                                                                                                                                                                            if "usd_24h_change" in coin_data:
                                                                                                                                                                                                            cleaned["price_change_24h"] = float(coin_data["usd_24h_change"])
                                                                                                                                                                                                        break

                                                                                                                                                                                                        # Validate and sanitize values
                                                                                                                                                                                                            for key, value in cleaned.items():
                                                                                                                                                                                                                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                                                                                                                                                                                                                cleaned[key] = 0.0
                                                                                                                                                                                                                    elif value < 0 and key in ["price", "volume_24h", "market_cap"]:
                                                                                                                                                                                                                    cleaned[key] = 0.0

                                                                                                                                                                                                                return cleaned

                                                                                                                                                                                                                    def _calculate_technical_indicators(self, symbol: str, data: Dict[str, Any]) -> TechnicalIndicators:
                                                                                                                                                                                                                    """Calculate technical indicators from price data."""
                                                                                                                                                                                                                    indicators = TechnicalIndicators()

                                                                                                                                                                                                                    # Get price history
                                                                                                                                                                                                                    current_price = data.get("price", 0.0)
                                                                                                                                                                                                                        if current_price > 0:
                                                                                                                                                                                                                        self._update_price_history(symbol, current_price)
                                                                                                                                                                                                                        self._update_volume_history(symbol, data.get("volume_24h", 0.0))

                                                                                                                                                                                                                        price_history = self.price_history.get(symbol, [])
                                                                                                                                                                                                                        volume_history = self.volume_history.get(symbol, [])

                                                                                                                                                                                                                            if len(price_history) < 2:
                                                                                                                                                                                                                        return indicators

                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                            prices = np.array(price_history)
                                                                                                                                                                                                                            volumes = np.array(volume_history) if volume_history else np.ones_like(prices)

                                                                                                                                                                                                                            # RSI calculation
                                                                                                                                                                                                                                if len(prices) >= 14:
                                                                                                                                                                                                                                indicators.rsi_14 = self._calculate_rsi(prices, 14)
                                                                                                                                                                                                                                    if len(prices) >= 21:
                                                                                                                                                                                                                                    indicators.rsi_21 = self._calculate_rsi(prices, 21)

                                                                                                                                                                                                                                    # MACD calculation
                                                                                                                                                                                                                                        if len(prices) >= 26:
                                                                                                                                                                                                                                        macd_result = self._calculate_macd(prices)
                                                                                                                                                                                                                                        indicators.macd_line, indicators.macd_signal, indicators.macd_histogram = macd_result

                                                                                                                                                                                                                                        # Bollinger Bands
                                                                                                                                                                                                                                            if len(prices) >= 20:
                                                                                                                                                                                                                                            bb_result = self._calculate_bollinger_bands(prices)
                                                                                                                                                                                                                                            (
                                                                                                                                                                                                                                            indicators.bb_upper,
                                                                                                                                                                                                                                            indicators.bb_middle,
                                                                                                                                                                                                                                            indicators.bb_lower,
                                                                                                                                                                                                                                            indicators.bb_position,
                                                                                                                                                                                                                                            ) = bb_result

                                                                                                                                                                                                                                            # Moving averages
                                                                                                                                                                                                                                                if len(prices) >= 12:
                                                                                                                                                                                                                                                indicators.ema_12 = self._calculate_ema(prices, 12)
                                                                                                                                                                                                                                                    if len(prices) >= 20:
                                                                                                                                                                                                                                                    indicators.sma_20 = np.mean(prices[-20:])
                                                                                                                                                                                                                                                        if len(prices) >= 26:
                                                                                                                                                                                                                                                        indicators.ema_26 = self._calculate_ema(prices, 26)

                                                                                                                                                                                                                                                        # Volume indicators
                                                                                                                                                                                                                                                            if len(volumes) >= 20:
                                                                                                                                                                                                                                                            indicators.volume_sma_20 = np.mean(volumes[-20:])
                                                                                                                                                                                                                                                                if indicators.volume_sma_20 > 0:
                                                                                                                                                                                                                                                                indicators.volume_ratio = volumes[-1] / indicators.volume_sma_20

                                                                                                                                                                                                                                                                # ATR calculation
                                                                                                                                                                                                                                                                    if len(prices) >= 14:
                                                                                                                                                                                                                                                                    indicators.atr_14 = self._calculate_atr(prices, 14)

                                                                                                                                                                                                                                                                    # Stochastic oscillator
                                                                                                                                                                                                                                                                        if len(prices) >= 14:
                                                                                                                                                                                                                                                                        stoch_result = self._calculate_stochastic(prices, 14)
                                                                                                                                                                                                                                                                        indicators.stoch_k, indicators.stoch_d = stoch_result

                                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                            logger.error("Error calculating technical indicators for {0}: {1}".format(symbol, e))

                                                                                                                                                                                                                                                                        return indicators

                                                                                                                                                                                                                                                                            def _extract_onchain_metrics(self, raw_data: Dict[str, Any]) -> OnChainMetrics:
                                                                                                                                                                                                                                                                            """Extract on-chain metrics from Glassnode and other sources."""
                                                                                                                                                                                                                                                                            metrics = OnChainMetrics()

                                                                                                                                                                                                                                                                            # Extract from Glassnode data
                                                                                                                                                                                                                                                                                if "glassnode" in raw_data:
                                                                                                                                                                                                                                                                                gn_data = raw_data["glassnode"]["data"]
                                                                                                                                                                                                                                                                                latest_values = gn_data.get("latest_values", {})

                                                                                                                                                                                                                                                                                metrics.hash_rate = latest_values.get("hash_rate_mean", 0.0)
                                                                                                                                                                                                                                                                                metrics.active_addresses = int(latest_values.get("active_count", 0))
                                                                                                                                                                                                                                                                                metrics.transaction_count = int(latest_values.get("count", 0))
                                                                                                                                                                                                                                                                                metrics.mvrv_ratio = latest_values.get("mvrv", 1.0)
                                                                                                                                                                                                                                                                                metrics.nvt_ratio = latest_values.get("nvt", 50.0)
                                                                                                                                                                                                                                                                                metrics.sopr = latest_values.get("sopr", 1.0)

                                                                                                                                                                                                                                                                                # Calculate network health score
                                                                                                                                                                                                                                                                                composite_scores = gn_data.get("composite_scores", {})
                                                                                                                                                                                                                                                                                metrics.network_health_score = composite_scores.get("overall_score", 50.0)

                                                                                                                                                                                                                                                                                # Extract whale activity from Whale Alert
                                                                                                                                                                                                                                                                                    if "whale_alert" in raw_data:
                                                                                                                                                                                                                                                                                    wa_data = raw_data["whale_alert"]["data"]
                                                                                                                                                                                                                                                                                    # Process whale transactions (simplified)
                                                                                                                                                                                                                                                                                        if "transactions" in wa_data:
                                                                                                                                                                                                                                                                                        whale_count = len(wa_data["transactions"])
                                                                                                                                                                                                                                                                                        metrics.whale_activity_score = min(100.0, whale_count * 10.0)

                                                                                                                                                                                                                                                                                    return metrics

                                                                                                                                                                                                                                                                                        def _calculate_market_sentiment(self, raw_data: Dict[str, Any]) -> MarketSentiment:
                                                                                                                                                                                                                                                                                        """Calculate market sentiment from various sources."""
                                                                                                                                                                                                                                                                                        sentiment = MarketSentiment()

                                                                                                                                                                                                                                                                                        # Fear & Greed Index
                                                                                                                                                                                                                                                                                            if "fear_greed" in raw_data:
                                                                                                                                                                                                                                                                                            fg_data = raw_data["fear_greed"]["data"]
                                                                                                                                                                                                                                                                                                if "value" in fg_data:
                                                                                                                                                                                                                                                                                                sentiment.fear_greed_index = float(fg_data["value"])
                                                                                                                                                                                                                                                                                                sentiment.fear_greed_classification = fg_data.get("value_classification", "Neutral")

                                                                                                                                                                                                                                                                                                # CoinGecko sentiment
                                                                                                                                                                                                                                                                                                    if "coingecko" in raw_data:
                                                                                                                                                                                                                                                                                                    cg_data = raw_data["coingecko"]["data"]
                                                                                                                                                                                                                                                                                                    market_sentiment_data = cg_data.get("market_sentiment", {})
                                                                                                                                                                                                                                                                                                    sentiment.trending_score = float(market_sentiment_data.get("score", 0))

                                                                                                                                                                                                                                                                                                    # Market dominance
                                                                                                                                                                                                                                                                                                    dominance = cg_data.get("market_dominance", {})
                                                                                                                                                                                                                                                                                                        if "btc" in dominance:
                                                                                                                                                                                                                                                                                                        sentiment.market_dominance = float(dominance["btc"])

                                                                                                                                                                                                                                                                                                    return sentiment

                                                                                                                                                                                                                                                                                                        def _calculate_derived_metrics(self, data: Dict[str, Any], indicators: TechnicalIndicators) -> Dict[str, float]:
                                                                                                                                                                                                                                                                                                        """Calculate derived metrics for the trading system."""
                                                                                                                                                                                                                                                                                                        metrics = {
                                                                                                                                                                                                                                                                                                        "volatility": 0.5,
                                                                                                                                                                                                                                                                                                        "trend_strength": 0.5,
                                                                                                                                                                                                                                                                                                        "entropy_level": 4.0,
                                                                                                                                                                                                                                                                                                        "liquidity_score": 0.5,
                                                                                                                                                                                                                                                                                                        "momentum_score": 0.5,
                                                                                                                                                                                                                                                                                                        }

                                                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                                                            # Volatility from ATR
                                                                                                                                                                                                                                                                                                                if indicators.atr_14 > 0 and data.get("price", 0) > 0:
                                                                                                                                                                                                                                                                                                                metrics["volatility"] = min(1.0, indicators.atr_14 / data["price"])

                                                                                                                                                                                                                                                                                                                # Trend strength from MACD and RSI
                                                                                                                                                                                                                                                                                                                    if indicators.macd_histogram != 0:
                                                                                                                                                                                                                                                                                                                    macd_strength = abs(indicators.macd_histogram) / max(abs(indicators.macd_line), 1.0)
                                                                                                                                                                                                                                                                                                                    rsi_trend = abs(indicators.rsi_14 - 50) / 50.0
                                                                                                                                                                                                                                                                                                                    metrics["trend_strength"] = min(1.0, (macd_strength + rsi_trend) / 2.0)

                                                                                                                                                                                                                                                                                                                    # Entropy level (market, uncertainty)
                                                                                                                                                                                                                                                                                                                    volatility = metrics["volatility"]
                                                                                                                                                                                                                                                                                                                    bb_width = abs(indicators.bb_upper - indicators.bb_lower) / max(indicators.bb_middle, 1.0)
                                                                                                                                                                                                                                                                                                                    metrics["entropy_level"] = min(10.0, 2.0 + volatility * 4.0 + bb_width * 4.0)

                                                                                                                                                                                                                                                                                                                    # Liquidity score from volume
                                                                                                                                                                                                                                                                                                                        if indicators.volume_ratio > 0:
                                                                                                                                                                                                                                                                                                                        metrics["liquidity_score"] = min(1.0, np.log(indicators.volume_ratio + 1) / 2.0)

                                                                                                                                                                                                                                                                                                                        # Momentum score
                                                                                                                                                                                                                                                                                                                            if indicators.rsi_14 > 0:
                                                                                                                                                                                                                                                                                                                            rsi_momentum = (indicators.rsi_14 - 50) / 50.0
                                                                                                                                                                                                                                                                                                                            macd_momentum = indicators.macd_line / max(abs(indicators.macd_line), 1.0)
                                                                                                                                                                                                                                                                                                                            metrics["momentum_score"] = (rsi_momentum + macd_momentum) / 2.0
                                                                                                                                                                                                                                                                                                                            metrics["momentum_score"] = max(-1.0, min(1.0, metrics["momentum_score"]))

                                                                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                                                                logger.error("Error calculating derived metrics: {0}".format(e))

                                                                                                                                                                                                                                                                                                                            return metrics

                                                                                                                                                                                                                                                                                                                                def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
                                                                                                                                                                                                                                                                                                                                """Calculate RSI indicator."""
                                                                                                                                                                                                                                                                                                                                    if len(prices) < period + 1:
                                                                                                                                                                                                                                                                                                                                return 50.0

                                                                                                                                                                                                                                                                                                                                deltas = np.diff(prices)
                                                                                                                                                                                                                                                                                                                                gains = np.where(deltas > 0, deltas, 0)
                                                                                                                                                                                                                                                                                                                                losses = np.where(deltas < 0, -deltas, 0)

                                                                                                                                                                                                                                                                                                                                avg_gain = np.mean(gains[-period:])
                                                                                                                                                                                                                                                                                                                                avg_loss = np.mean(losses[-period:])

                                                                                                                                                                                                                                                                                                                                    if avg_loss == 0:
                                                                                                                                                                                                                                                                                                                                return 100.0

                                                                                                                                                                                                                                                                                                                                rs = avg_gain / avg_loss
                                                                                                                                                                                                                                                                                                                                rsi = 100 - (100 / (1 + rs))

                                                                                                                                                                                                                                                                                                                            return float(rsi)

                                                                                                                                                                                                                                                                                                                                def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
                                                                                                                                                                                                                                                                                                                                """Calculate MACD indicator."""
                                                                                                                                                                                                                                                                                                                                    if len(prices) < 26:
                                                                                                                                                                                                                                                                                                                                return 0.0, 0.0, 0.0

                                                                                                                                                                                                                                                                                                                                ema12 = self._calculate_ema(prices, 12)
                                                                                                                                                                                                                                                                                                                                ema26 = self._calculate_ema(prices, 26)
                                                                                                                                                                                                                                                                                                                                macd_line = ema12 - ema26

                                                                                                                                                                                                                                                                                                                                # Simple signal line (9-period EMA of MACD)
                                                                                                                                                                                                                                                                                                                                macd_signal = macd_line * 0.9  # Simplified
                                                                                                                                                                                                                                                                                                                                macd_histogram = macd_line - macd_signal

                                                                                                                                                                                                                                                                                                                            return float(macd_line), float(macd_signal), float(macd_histogram)

                                                                                                                                                                                                                                                                                                                                def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
                                                                                                                                                                                                                                                                                                                                """Calculate Exponential Moving Average."""
                                                                                                                                                                                                                                                                                                                                    if len(prices) < period:
                                                                                                                                                                                                                                                                                                                                return float(np.mean(prices))

                                                                                                                                                                                                                                                                                                                                alpha = 2 / (period + 1)
                                                                                                                                                                                                                                                                                                                                ema = prices[0]

                                                                                                                                                                                                                                                                                                                                    for price in prices[1:]:
                                                                                                                                                                                                                                                                                                                                    ema = alpha * price + (1 - alpha) * ema

                                                                                                                                                                                                                                                                                                                                return float(ema)

                                                                                                                                                                                                                                                                                                                                    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float, float, float]:
                                                                                                                                                                                                                                                                                                                                    """Calculate Bollinger Bands."""
                                                                                                                                                                                                                                                                                                                                        if len(prices) < period:
                                                                                                                                                                                                                                                                                                                                    return 0.0, 0.0, 0.0, 0.5

                                                                                                                                                                                                                                                                                                                                    sma = np.mean(prices[-period:])
                                                                                                                                                                                                                                                                                                                                    std = np.std(prices[-period:])

                                                                                                                                                                                                                                                                                                                                    upper = sma + (2 * std)
                                                                                                                                                                                                                                                                                                                                    lower = sma - (2 * std)

                                                                                                                                                                                                                                                                                                                                    current_price = prices[-1]
                                                                                                                                                                                                                                                                                                                                    position = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
                                                                                                                                                                                                                                                                                                                                    position = max(0.0, min(1.0, position))

                                                                                                                                                                                                                                                                                                                                return float(upper), float(sma), float(lower), float(position)

                                                                                                                                                                                                                                                                                                                                    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
                                                                                                                                                                                                                                                                                                                                    """Calculate Average True Range."""
                                                                                                                                                                                                                                                                                                                                        if len(prices) < period + 1:
                                                                                                                                                                                                                                                                                                                                    return 0.0

                                                                                                                                                                                                                                                                                                                                    # Simplified ATR using price differences
                                                                                                                                                                                                                                                                                                                                    true_ranges = np.abs(np.diff(prices))
                                                                                                                                                                                                                                                                                                                                    atr = np.mean(true_ranges[-period:])

                                                                                                                                                                                                                                                                                                                                return float(atr)

                                                                                                                                                                                                                                                                                                                                    def _calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
                                                                                                                                                                                                                                                                                                                                    """Calculate Stochastic Oscillator."""
                                                                                                                                                                                                                                                                                                                                        if len(prices) < period:
                                                                                                                                                                                                                                                                                                                                    return 50.0, 50.0

                                                                                                                                                                                                                                                                                                                                    recent_prices = prices[-period:]
                                                                                                                                                                                                                                                                                                                                    highest_high = np.max(recent_prices)
                                                                                                                                                                                                                                                                                                                                    lowest_low = np.min(recent_prices)
                                                                                                                                                                                                                                                                                                                                    current_price = prices[-1]

                                                                                                                                                                                                                                                                                                                                        if highest_high == lowest_low:
                                                                                                                                                                                                                                                                                                                                        k_percent = 50.0
                                                                                                                                                                                                                                                                                                                                            else:
                                                                                                                                                                                                                                                                                                                                            k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100

                                                                                                                                                                                                                                                                                                                                            # Simplified D% (3-period SMA of K%)
                                                                                                                                                                                                                                                                                                                                            d_percent = k_percent * 0.8  # Simplified

                                                                                                                                                                                                                                                                                                                                        return float(k_percent), float(d_percent)

                                                                                                                                                                                                                                                                                                                                            def _assess_data_quality(self, raw_data: Dict[str, Any]) -> Tuple[DataQuality, float]:
                                                                                                                                                                                                                                                                                                                                            """Assess the quality of fetched data."""
                                                                                                                                                                                                                                                                                                                                            source_count = len(raw_data)
                                                                                                                                                                                                                                                                                                                                            total_sources = len(self.handlers)

                                                                                                                                                                                                                                                                                                                                                if source_count == 0:
                                                                                                                                                                                                                                                                                                                                            return DataQuality.FAILED, 0.0

                                                                                                                                                                                                                                                                                                                                            # Calculate freshness score
                                                                                                                                                                                                                                                                                                                                            current_time = time.time()
                                                                                                                                                                                                                                                                                                                                            freshness_scores = []

                                                                                                                                                                                                                                                                                                                                                for source_data in raw_data.values():
                                                                                                                                                                                                                                                                                                                                                timestamp = source_data.get("timestamp", current_time)
                                                                                                                                                                                                                                                                                                                                                age = current_time - timestamp
                                                                                                                                                                                                                                                                                                                                                freshness = max(0.0, 1.0 - (age / 3600))  # 1 hour decay
                                                                                                                                                                                                                                                                                                                                                freshness_scores.append(freshness)

                                                                                                                                                                                                                                                                                                                                                avg_freshness = np.mean(freshness_scores) if freshness_scores else 0.0
                                                                                                                                                                                                                                                                                                                                                source_ratio = source_count / total_sources

                                                                                                                                                                                                                                                                                                                                                # Determine quality level
                                                                                                                                                                                                                                                                                                                                                    if source_ratio >= 0.8 and avg_freshness >= 0.9:
                                                                                                                                                                                                                                                                                                                                                    quality = DataQuality.EXCELLENT
                                                                                                                                                                                                                                                                                                                                                        elif source_ratio >= 0.6 and avg_freshness >= 0.7:
                                                                                                                                                                                                                                                                                                                                                        quality = DataQuality.GOOD
                                                                                                                                                                                                                                                                                                                                                            elif source_ratio >= 0.4 and avg_freshness >= 0.5:
                                                                                                                                                                                                                                                                                                                                                            quality = DataQuality.ACCEPTABLE
                                                                                                                                                                                                                                                                                                                                                                elif source_ratio >= 0.2:
                                                                                                                                                                                                                                                                                                                                                                quality = DataQuality.POOR
                                                                                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                                                                    quality = DataQuality.FAILED

                                                                                                                                                                                                                                                                                                                                                                return quality, avg_freshness

                                                                                                                                                                                                                                                                                                                                                                    def _calculate_quality_score(self, quality: DataQuality, freshness: float) -> float:
                                                                                                                                                                                                                                                                                                                                                                    """Calculate a numeric quality score."""
                                                                                                                                                                                                                                                                                                                                                                    quality_scores = {
                                                                                                                                                                                                                                                                                                                                                                    DataQuality.EXCELLENT: 1.0,
                                                                                                                                                                                                                                                                                                                                                                    DataQuality.GOOD: 0.8,
                                                                                                                                                                                                                                                                                                                                                                    DataQuality.ACCEPTABLE: 0.6,
                                                                                                                                                                                                                                                                                                                                                                    DataQuality.POOR: 0.4,
                                                                                                                                                                                                                                                                                                                                                                    DataQuality.FAILED: 0.0,
                                                                                                                                                                                                                                                                                                                                                                    }

                                                                                                                                                                                                                                                                                                                                                                    base_score = quality_scores[quality]
                                                                                                                                                                                                                                                                                                                                                                return (base_score + freshness) / 2.0

                                                                                                                                                                                                                                                                                                                                                                    def _is_cached(self, symbol: str) -> bool:
                                                                                                                                                                                                                                                                                                                                                                    """Check if symbol data is cached and fresh."""
                                                                                                                                                                                                                                                                                                                                                                        if symbol not in self.data_cache:
                                                                                                                                                                                                                                                                                                                                                                    return False

                                                                                                                                                                                                                                                                                                                                                                        if symbol not in self.cache_expiry:
                                                                                                                                                                                                                                                                                                                                                                    return False

                                                                                                                                                                                                                                                                                                                                                                return time.time() < self.cache_expiry[symbol]

                                                                                                                                                                                                                                                                                                                                                                    def _cache_data(self, symbol: str, data: MarketDataPacket) -> None:
                                                                                                                                                                                                                                                                                                                                                                    """Cache market data packet."""
                                                                                                                                                                                                                                                                                                                                                                    self.data_cache[symbol] = data
                                                                                                                                                                                                                                                                                                                                                                    self.cache_expiry[symbol] = time.time() + self.config["cache_ttl"]

                                                                                                                                                                                                                                                                                                                                                                        def _update_price_history(self, symbol: str, price: float) -> None:
                                                                                                                                                                                                                                                                                                                                                                        """Update price history for technical analysis."""
                                                                                                                                                                                                                                                                                                                                                                            if symbol not in self.price_history:
                                                                                                                                                                                                                                                                                                                                                                            self.price_history[symbol] = []

                                                                                                                                                                                                                                                                                                                                                                            self.price_history[symbol].append(price)

                                                                                                                                                                                                                                                                                                                                                                            # Limit history size
                                                                                                                                                                                                                                                                                                                                                                            max_history = self.config.get("max_price_history", 1000)
                                                                                                                                                                                                                                                                                                                                                                                if len(self.price_history[symbol]) > max_history:
                                                                                                                                                                                                                                                                                                                                                                                self.price_history[symbol] = self.price_history[symbol][-max_history:]

                                                                                                                                                                                                                                                                                                                                                                                    def _update_volume_history(self, symbol: str, volume: float) -> None:
                                                                                                                                                                                                                                                                                                                                                                                    """Update volume history for analysis."""
                                                                                                                                                                                                                                                                                                                                                                                        if symbol not in self.volume_history:
                                                                                                                                                                                                                                                                                                                                                                                        self.volume_history[symbol] = []

                                                                                                                                                                                                                                                                                                                                                                                        self.volume_history[symbol].append(volume)

                                                                                                                                                                                                                                                                                                                                                                                        # Limit history size
                                                                                                                                                                                                                                                                                                                                                                                        max_history = self.config.get("max_price_history", 1000)
                                                                                                                                                                                                                                                                                                                                                                                            if len(self.volume_history[symbol]) > max_history:
                                                                                                                                                                                                                                                                                                                                                                                            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]

                                                                                                                                                                                                                                                                                                                                                                                                def _update_metrics(self, success: bool, latency: float) -> None:
                                                                                                                                                                                                                                                                                                                                                                                                """Update pipeline performance metrics."""
                                                                                                                                                                                                                                                                                                                                                                                                self.metrics.total_requests += 1

                                                                                                                                                                                                                                                                                                                                                                                                    if success:
                                                                                                                                                                                                                                                                                                                                                                                                    self.metrics.successful_requests += 1
                                                                                                                                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                                                                                                                        self.metrics.failed_requests += 1

                                                                                                                                                                                                                                                                                                                                                                                                        # Update average latency
                                                                                                                                                                                                                                                                                                                                                                                                        total_latency = self.metrics.average_latency * (self.metrics.total_requests - 1) + latency
                                                                                                                                                                                                                                                                                                                                                                                                        self.metrics.average_latency = total_latency / self.metrics.total_requests

                                                                                                                                                                                                                                                                                                                                                                                                        # Update success rate
                                                                                                                                                                                                                                                                                                                                                                                                        success_rate = self.metrics.successful_requests / self.metrics.total_requests
                                                                                                                                                                                                                                                                                                                                                                                                        self.metrics.uptime_percentage = success_rate * 100

                                                                                                                                                                                                                                                                                                                                                                                                        self.metrics.last_update = time.time()

                                                                                                                                                                                                                                                                                                                                                                                                            def _log_to_registry(self, packet: MarketDataPacket) -> None:
                                                                                                                                                                                                                                                                                                                                                                                                            """Log market data packet to registry for tracking."""
                                                                                                                                                                                                                                                                                                                                                                                                                if not self.registry:
                                                                                                                                                                                                                                                                                                                                                                                                            return

                                                                                                                                                                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                                                                                                                                                                # Create registry entry for market data
                                                                                                                                                                                                                                                                                                                                                                                                                schwafit_info = {
                                                                                                                                                                                                                                                                                                                                                                                                                "symbol": packet.symbol,
                                                                                                                                                                                                                                                                                                                                                                                                                "price": packet.price,
                                                                                                                                                                                                                                                                                                                                                                                                                "rsi_14": packet.technical_indicators.rsi_14,
                                                                                                                                                                                                                                                                                                                                                                                                                "macd_line": packet.technical_indicators.macd_line,
                                                                                                                                                                                                                                                                                                                                                                                                                "bb_position": packet.technical_indicators.bb_position,
                                                                                                                                                                                                                                                                                                                                                                                                                "volatility": packet.volatility,
                                                                                                                                                                                                                                                                                                                                                                                                                "trend_strength": packet.trend_strength,
                                                                                                                                                                                                                                                                                                                                                                                                                "data_quality": packet.data_quality.value,
                                                                                                                                                                                                                                                                                                                                                                                                                "source_count": packet.source_count,
                                                                                                                                                                                                                                                                                                                                                                                                                "fear_greed": packet.market_sentiment.fear_greed_index,
                                                                                                                                                                                                                                                                                                                                                                                                                "network_health": packet.onchain_metrics.network_health_score,
                                                                                                                                                                                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                                                                                                                                                                                # Log as market data trigger
                                                                                                                                                                                                                                                                                                                                                                                                                self.registry.log_trigger(
                                                                                                                                                                                                                                                                                                                                                                                                                asset=packet.symbol,
                                                                                                                                                                                                                                                                                                                                                                                                                phase=packet.technical_indicators.bb_position,
                                                                                                                                                                                                                                                                                                                                                                                                                drift=packet.momentum_score,
                                                                                                                                                                                                                                                                                                                                                                                                                schwafit_info=schwafit_info,
                                                                                                                                                                                                                                                                                                                                                                                                                trade_result={
                                                                                                                                                                                                                                                                                                                                                                                                                "market_data_update": True,
                                                                                                                                                                                                                                                                                                                                                                                                                "timestamp": packet.timestamp,
                                                                                                                                                                                                                                                                                                                                                                                                                "data_quality_score": packet.metadata.get("quality_score", 0.0),
                                                                                                                                                                                                                                                                                                                                                                                                                },
                                                                                                                                                                                                                                                                                                                                                                                                                )

                                                                                                                                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                    logger.error("Failed to log market data to registry: {0}".format(e))

                                                                                                                                                                                                                                                                                                                                                                                                                        def _create_fallback_packet(self, symbol: str) -> MarketDataPacket:
                                                                                                                                                                                                                                                                                                                                                                                                                        """Create a fallback market data packet when all sources fail."""
                                                                                                                                                                                                                                                                                                                                                                                                                    return MarketDataPacket(
                                                                                                                                                                                                                                                                                                                                                                                                                    symbol=symbol,
                                                                                                                                                                                                                                                                                                                                                                                                                    price=0.0,
                                                                                                                                                                                                                                                                                                                                                                                                                    volume_24h=0.0,
                                                                                                                                                                                                                                                                                                                                                                                                                    market_cap=0.0,
                                                                                                                                                                                                                                                                                                                                                                                                                    price_change_24h=0.0,
                                                                                                                                                                                                                                                                                                                                                                                                                    volume_change_24h=0.0,
                                                                                                                                                                                                                                                                                                                                                                                                                    timestamp=time.time(),
                                                                                                                                                                                                                                                                                                                                                                                                                    data_quality=DataQuality.FAILED,
                                                                                                                                                                                                                                                                                                                                                                                                                    source_count=0,
                                                                                                                                                                                                                                                                                                                                                                                                                    freshness_score=0.0,
                                                                                                                                                                                                                                                                                                                                                                                                                    technical_indicators=TechnicalIndicators(),
                                                                                                                                                                                                                                                                                                                                                                                                                    onchain_metrics=OnChainMetrics(),
                                                                                                                                                                                                                                                                                                                                                                                                                    market_sentiment=MarketSentiment(),
                                                                                                                                                                                                                                                                                                                                                                                                                    sources_used=[],
                                                                                                                                                                                                                                                                                                                                                                                                                    api_latencies={},
                                                                                                                                                                                                                                                                                                                                                                                                                    metadata={"fallback": True, "error": "All data sources failed"},
                                                                                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                                                                                        def get_pipeline_status(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                                                                                                                                                                        """Get comprehensive pipeline status and metrics."""
                                                                                                                                                                                                                                                                                                                                                                                                                        uptime = time.time() - self.start_time

                                                                                                                                                                                                                                                                                                                                                                                                                    return {
                                                                                                                                                                                                                                                                                                                                                                                                                    "pipeline_status": "running" if self.handlers else "stopped",
                                                                                                                                                                                                                                                                                                                                                                                                                    "uptime_seconds": uptime,
                                                                                                                                                                                                                                                                                                                                                                                                                    "active_handlers": list(self.handlers.keys()),
                                                                                                                                                                                                                                                                                                                                                                                                                    "cached_symbols": list(self.data_cache.keys()),
                                                                                                                                                                                                                                                                                                                                                                                                                    "metrics": {
                                                                                                                                                                                                                                                                                                                                                                                                                    "total_requests": self.metrics.total_requests,
                                                                                                                                                                                                                                                                                                                                                                                                                    "success_rate": self.metrics.successful_requests / max(1, self.metrics.total_requests),
                                                                                                                                                                                                                                                                                                                                                                                                                    "average_latency": self.metrics.average_latency,
                                                                                                                                                                                                                                                                                                                                                                                                                    "uptime_percentage": self.metrics.uptime_percentage,
                                                                                                                                                                                                                                                                                                                                                                                                                    "cache_size": len(self.data_cache),
                                                                                                                                                                                                                                                                                                                                                                                                                    },
                                                                                                                                                                                                                                                                                                                                                                                                                    "config": self.config,
                                                                                                                                                                                                                                                                                                                                                                                                                    }

                                                                                                                                                                                                                                                                                                                                                                                                                        async def health_check(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                                                                                                                                                                        """Perform a health check on all API handlers."""
                                                                                                                                                                                                                                                                                                                                                                                                                        health_status = {}

                                                                                                                                                                                                                                                                                                                                                                                                                            for handler_name, handler in self.handlers.items():
                                                                                                                                                                                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                                                                                                                                                                                start_time = time.time()
                                                                                                                                                                                                                                                                                                                                                                                                                                # Attempt a simple data fetch
                                                                                                                                                                                                                                                                                                                                                                                                                                test_data = await handler.get_data(force_refresh=False)
                                                                                                                                                                                                                                                                                                                                                                                                                                latency = time.time() - start_time

                                                                                                                                                                                                                                                                                                                                                                                                                                health_status[handler_name] = {
                                                                                                                                                                                                                                                                                                                                                                                                                                "status": "healthy" if test_data else "unhealthy",
                                                                                                                                                                                                                                                                                                                                                                                                                                "latency": latency,
                                                                                                                                                                                                                                                                                                                                                                                                                                "last_check": time.time(),
                                                                                                                                                                                                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                    health_status[handler_name] = {
                                                                                                                                                                                                                                                                                                                                                                                                                                    "status": "error",
                                                                                                                                                                                                                                                                                                                                                                                                                                    "error": str(e),
                                                                                                                                                                                                                                                                                                                                                                                                                                    "last_check": time.time(),
                                                                                                                                                                                                                                                                                                                                                                                                                                    }

                                                                                                                                                                                                                                                                                                                                                                                                                                return health_status


                                                                                                                                                                                                                                                                                                                                                                                                                                # Factory function
                                                                                                                                                                                                                                                                                                                                                                                                                                    def create_unified_pipeline(config: Optional[Dict[str, Any]] = None) -> UnifiedMarketDataPipeline:
                                                                                                                                                                                                                                                                                                                                                                                                                                    """Create a new unified market data pipeline instance."""
                                                                                                                                                                                                                                                                                                                                                                                                                                return UnifiedMarketDataPipeline(config)


                                                                                                                                                                                                                                                                                                                                                                                                                                # Example usage
                                                                                                                                                                                                                                                                                                                                                                                                                                    async def demo_pipeline():
                                                                                                                                                                                                                                                                                                                                                                                                                                    """Demonstrate the unified market data pipeline."""
                                                                                                                                                                                                                                                                                                                                                                                                                                    pipeline = create_unified_pipeline()

                                                                                                                                                                                                                                                                                                                                                                                                                                    # Get market data for Bitcoin
                                                                                                                                                                                                                                                                                                                                                                                                                                    btc_data = await pipeline.get_market_data("BTC")

                                                                                                                                                                                                                                                                                                                                                                                                                                    print("BTC Price: ${0:,.2f}".format(btc_data.price))
                                                                                                                                                                                                                                                                                                                                                                                                                                    print("RSI: {0:.2f}".format(btc_data.technical_indicators.rsi_14))
                                                                                                                                                                                                                                                                                                                                                                                                                                    print("Data Quality: {0}".format(btc_data.data_quality.value))
                                                                                                                                                                                                                                                                                                                                                                                                                                    print("Sources Used: {0}".format(', '.join(btc_data.sources_used)))

                                                                                                                                                                                                                                                                                                                                                                                                                                    # Health check
                                                                                                                                                                                                                                                                                                                                                                                                                                    health = await pipeline.health_check()
                                                                                                                                                                                                                                                                                                                                                                                                                                    print("API Health: {0}".format(health))


                                                                                                                                                                                                                                                                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                                                                                        asyncio.run(demo_pipeline())
