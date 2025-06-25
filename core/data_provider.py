# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Data Provider - Schwabot Data Sourcing and Distribution System
=============================================================

Provides unified data sourcing, normalization, and distribution for the
Schwabot trading system. Handles multiple data sources, real-time feeds,
and historical data with mathematical integration.

Features:
- Multiple data source integration (APIs, WebSockets, databases)
- Real-time data normalization and validation
- Historical data management and caching
- Data quality monitoring and validation
- Mathematical data processing integration
- Event-driven data distribution
"""

import asyncio
import json
import logging
# from core.unified_math_system import unified_math  # F811: duplicate import
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import hashlib

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources."""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    DATABASE = "database"
    FILE = "file"
    SIMULATOR = "simulator"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class DataPoint:
    """Single data point with metadata."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str = ""
    quality: DataQuality = DataQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataStream:
    """Data stream configuration."""
    stream_id: str
    source_type: DataSourceType
    symbols: List[str]
    update_interval: float
    is_active: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    completeness: float  # 0.0 to 1.0
    accuracy: float  # 0.0 to 1.0
    timeliness: float  # 0.0 to 1.0
    consistency: float  # 0.0 to 1.0
    overall_score: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataProviderConfig:
    """Data provider configuration."""
    max_cache_size: int = 10000
    cache_ttl: float = 300.0  # seconds
    quality_threshold: float = 0.7
    enable_validation: bool = True
    enable_caching: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


class DataProvider:
    """
    Comprehensive data provider system for Schwabot.

    Handles data sourcing, normalization, validation, and distribution
    with mathematical integration for trading operations.
    """

    def __init__(self, config: Optional[DataProviderConfig] = None):
        """Initialize data provider."""
        self.config = config or DataProviderConfig()

        # Core data structures
        self.data_streams: Dict[str, DataStream] = {}
        self.data_cache: Dict[str, List[DataPoint]] = {}
        self.subscribers: Dict[str, List[Callable[[DataPoint], None]]] = {}
        self.quality_metrics: Dict[str, DataQualityMetrics] = {}

        # Performance tracking
        self.performance_stats = {
            "total_data_points": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "quality_violations": 0,
            "last_update": datetime.now()
        }

        # Threading
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None

        # Initialize default streams
        self._initialize_default_streams()

        logger.info("Data Provider initialized")

    def _initialize_default_streams(self) -> None:
        """Initialize default data streams."""
        default_streams = [
            DataStream(
                stream_id="btc_live",
                source_type=DataSourceType.SIMULATOR,
                symbols=["BTC/USD"],
                update_interval=1.0,
                config={"base_price": 50000.0, "volatility": 0.02}
            ),
            DataStream(
                stream_id="eth_live",
                source_type=DataSourceType.SIMULATOR,
                symbols=["ETH/USD"],
                update_interval=1.0,
                config={"base_price": 3000.0, "volatility": 0.025}
            ),
            DataStream(
                stream_id="xrp_live",
                source_type=DataSourceType.SIMULATOR,
                symbols=["XRP/USD"],
                update_interval=1.0,
                config={"base_price": 0.5, "volatility": 0.03}
            )
        ]

        for stream in default_streams:
            self.add_data_stream(stream)

    def add_data_stream(self, stream: DataStream) -> bool:
        """Add a new data stream."""
        if stream.stream_id in self.data_streams:
            logger.warning(f"Stream {stream.stream_id} already exists. Overwriting.")

        self.data_streams[stream.stream_id] = stream

        # Initialize cache for stream
        for symbol in stream.symbols:
            cache_key = f"{stream.stream_id}_{symbol}"
            self.data_cache[cache_key] = []
            self.subscribers[cache_key] = []

        logger.info(f"Data stream added: {stream.stream_id} ({stream.source_type.value})")
        return True

    def remove_data_stream(self, stream_id: str) -> bool:
        """Remove a data stream."""
        if stream_id not in self.data_streams:
            logger.warning(f"Stream {stream_id} not found.")
            return False

        stream = self.data_streams[stream_id]

        # Clean up cache and subscribers
        for symbol in stream.symbols:
            cache_key = f"{stream_id}_{symbol}"
            if cache_key in self.data_cache:
                del self.data_cache[cache_key]
            if cache_key in self.subscribers:
                del self.subscribers[cache_key]

        del self.data_streams[stream_id]
        logger.info(f"Data stream removed: {stream_id}")
        return True

    def start_data_provider(self) -> bool:
        """Start data provider service."""
        self.is_running = True

        # Start update thread
        self.update_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        self.update_thread.start()

        logger.info("Data Provider started")
        return True

    def stop_data_provider(self) -> bool:
        """Stop data provider service."""
        self.is_running = False

        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)

        logger.info("Data Provider stopped")
        return True

    def _data_update_loop(self) -> None:
        """Main data update loop."""
        while self.is_running:
            try:
                # Update all active streams
                for stream_id, stream in self.data_streams.items():
                    if stream.is_active:
                        self._update_stream_data(stream)

                # Clean up old cache entries
                self._cleanup_cache()

                # Update performance stats
                self._update_performance_stats()

                # Sleep for a short interval
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Data update error: {e}")
                time.sleep(1.0)

    def _update_stream_data(self, stream: DataStream) -> None:
        """Update data for a specific stream."""
        current_time = datetime.now()

        for symbol in stream.symbols:
            # Generate or fetch data based on source type
            if stream.source_type == DataSourceType.SIMULATOR:
                data_point = self._generate_simulated_data(symbol, stream.config, current_time)
            else:
                # Placeholder for real data sources
                data_point = self._fetch_real_data(symbol, stream, current_time)

            # Validate data quality
            if self.config.enable_validation:
                data_point.quality = self._assess_data_quality(data_point)

            # Cache data point
            cache_key = f"{stream.stream_id}_{symbol}"
            self._cache_data_point(cache_key, data_point)

            # Notify subscribers
            self._notify_subscribers(cache_key, data_point)

            # Update quality metrics
            self._update_quality_metrics(cache_key, data_point)

    def _generate_simulated_data(self, symbol: str, config: Dict[str, Any],
                                timestamp: datetime) -> DataPoint:
        """Generate simulated market data."""
        base_price = config.get("base_price", 50000.0)
        volatility = config.get("volatility", 0.02)

        # Generate price with random walk
        price_change = np.random.normal(0, volatility)
        price = base_price * (1 + price_change)

        # Generate volume
        volume = np.random.uniform(100, 1000)

        # Generate bid/ask spread
        spread = price * 0.001  # 0.1% spread
        bid = price - spread / 2
        ask = price + spread / 2

        return DataPoint(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            source="simulator",
            metadata={"volatility": volatility, "base_price": base_price}
        )

    def _fetch_real_data(self, symbol: str, stream: DataStream,
                        timestamp: datetime) -> DataPoint:
        """Fetch real data from external sources (placeholder)."""
        # This would integrate with real data sources
        # For now, return simulated data
        return self._generate_simulated_data(symbol, stream.config, timestamp)

    def _assess_data_quality(self, data_point: DataPoint) -> DataQuality:
        """Assess data quality based on various metrics."""
        quality_score = 1.0

        # Check for missing values
        if data_point.price <= 0 or data_point.volume <= 0:
            quality_score -= 0.5

        # Check for extreme values
        if data_point.price > 1000000 or data_point.volume > 1000000:
            quality_score -= 0.3

        # Check timestamp freshness
        age = (datetime.now() - data_point.timestamp).total_seconds()
        if age > 60:  # More than 1 minute old
            quality_score -= 0.2

        # Determine quality level
        if quality_score >= 0.9:
            return DataQuality.EXCELLENT
        elif quality_score >= 0.7:
            return DataQuality.GOOD
        elif quality_score >= 0.5:
            return DataQuality.FAIR
        elif quality_score >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID

    def _cache_data_point(self, cache_key: str, data_point: DataPoint) -> None:
        """Cache a data point."""
        if cache_key not in self.data_cache:
            self.data_cache[cache_key] = []

        cache = self.data_cache[cache_key]
        cache.append(data_point)

        # Limit cache size
        if len(cache) > self.config.max_cache_size:
            cache.pop(0)

        self.performance_stats["total_data_points"] += 1

    def _notify_subscribers(self, cache_key: str, data_point: DataPoint) -> None:
        """Notify subscribers of new data."""
        if cache_key in self.subscribers:
            for subscriber in self.subscribers[cache_key]:
                try:
                    subscriber(data_point)
                except Exception as e:
                    logger.error(f"Subscriber notification error: {e}")

    def _update_quality_metrics(self, cache_key: str, data_point: DataPoint) -> None:
        """Update quality metrics for a data stream."""
        if cache_key not in self.quality_metrics:
            self.quality_metrics[cache_key] = DataQualityMetrics(
                completeness=1.0,
                accuracy=1.0,
                timeliness=1.0,
                consistency=1.0,
                overall_score=1.0
            )

        metrics = self.quality_metrics[cache_key]

        # Update metrics based on data point quality
        if data_point.quality == DataQuality.INVALID:
            metrics.accuracy *= 0.9
            self.performance_stats["quality_violations"] += 1

        # Recalculate overall score
        metrics.overall_score = (metrics.completeness + metrics.accuracy +
                               metrics.timeliness + metrics.consistency) / 4.0

    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.config.cache_ttl)

        for cache_key, cache in self.data_cache.items():
            # Remove old entries
            self.data_cache[cache_key] = [
                point for point in cache
                if point.timestamp > cutoff_time
            ]

    def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        self.performance_stats["last_update"] = datetime.now()

    def subscribe_to_data(self, symbol: str, stream_id: str,
                         callback: Callable[[DataPoint], None]) -> bool:
        """Subscribe to data updates for a specific symbol and stream."""
        cache_key = f"{stream_id}_{symbol}"

        if cache_key not in self.subscribers:
            self.subscribers[cache_key] = []

        if callback not in self.subscribers[cache_key]:
            self.subscribers[cache_key].append(callback)
            logger.info(f"New subscriber for {cache_key}: {callback.__name__}")
            return True

        return False

    def unsubscribe_from_data(self, symbol: str, stream_id: str,
                             callback: Callable[[DataPoint], None]) -> bool:
        """Unsubscribe from data updates."""
        cache_key = f"{stream_id}_{symbol}"

        if cache_key in self.subscribers and callback in self.subscribers[cache_key]:
            self.subscribers[cache_key].remove(callback)
            logger.info(f"Subscriber removed for {cache_key}: {callback.__name__}")
            return True

        return False

    def get_latest_data(self, symbol: str, stream_id: str) -> Optional[DataPoint]:
        """Get the latest data point for a symbol and stream."""
        cache_key = f"{stream_id}_{symbol}"

        if cache_key in self.data_cache and self.data_cache[cache_key]:
            return self.data_cache[cache_key][-1]

        return None

    def get_historical_data(self, symbol: str, stream_id: str,
                          hours: int = 24) -> List[DataPoint]:
        """Get historical data for a symbol and stream."""
        cache_key = f"{stream_id}_{symbol}"

        if cache_key not in self.data_cache:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            point for point in self.data_cache[cache_key]
            if point.timestamp > cutoff_time
        ]

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        return {
            "quality_metrics": {k: asdict(v) for k, v in self.quality_metrics.items()},
            "performance_stats": self.performance_stats,
            "active_streams": len([s for s in self.data_streams.values() if s.is_active]),
            "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "cache_size": sum(len(cache) for cache in self.data_cache.values())
        }

    def get_provider_status(self) -> Dict[str, Any]:
        """Get data provider status."""
        return {
            "is_running": self.is_running,
            "active_streams": len([s for s in self.data_streams.values() if s.is_active]),
            "total_streams": len(self.data_streams),
            "cache_entries": sum(len(cache) for cache in self.data_cache.values()),
            "subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "last_update": self.performance_stats["last_update"].isoformat()
        }


# Global data provider instance
data_provider = DataProvider()


def get_data_provider() -> DataProvider:
    """Get global data provider instance."""
    return data_provider


def main() -> None:
    """Main function for testing data provider."""
    logging.basicConfig(level=logging.INFO)

    safe_print("🧪 Testing Data Provider")
    safe_print("=" * 30)

    # Create data provider
    provider = DataProvider()

    # Start provider
    provider.start_data_provider()

    # Test data subscription
    def data_callback(data_point: DataPoint):
        safe_print(f"📊 Received data: {data_point.symbol} = ${data_point.price:.2f}")

    provider.subscribe_to_data("BTC/USD", "btc_live", data_callback)

    # Let it run for a few seconds
    time.sleep(5)

    # Get latest data
    latest_data = provider.get_latest_data("BTC/USD", "btc_live")
    if latest_data:
        safe_print(f"✅ Latest BTC price: ${latest_data.price:.2f}")

    # Get quality report
    quality_report = provider.get_data_quality_report()
    safe_print(f"📈 Quality report: {quality_report['performance_stats']['total_data_points']} data points")

    # Stop provider
    provider.stop_data_provider()

    safe_print("Data provider test completed!")


if __name__ == "__main__":
    main()
