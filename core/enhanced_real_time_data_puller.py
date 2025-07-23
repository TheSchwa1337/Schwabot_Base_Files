#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENHANCED REAL-TIME DATA PULLER - DYNAMIC ADAPTIVE DATA COLLECTION
====================================================================

Advanced real-time data pulling system that provides:
- Dynamic adaptive intervals based on market conditions
- Rolling measurements with time-weighted calculations
- Real-time timing triggers for data collection
- Integration with dynamic timing system
- Multi-source data aggregation with quality assessment
- Performance monitoring and optimization

Mathematical Foundation:
D(t) = {
    Pull Interval:  I_p(t) = adaptive_interval(volatility, momentum, regime)
    Data Quality:   Q_d(t) = quality_assessment(freshness, completeness, accuracy)
    Rolling Stats:  R_s(t) = time_weighted_average(data_series, decay_weights)
    Trigger Logic:  T_l(t) = trigger_function(market_conditions, thresholds)
}

Where:
- t: time parameter
- adaptive_interval: Dynamic interval calculation
- quality_assessment: Data quality evaluation
- time_weighted_average: Rolling statistics with exponential decay
- trigger_function: Timing trigger logic
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from collections import deque
import numpy as np

# Import dynamic timing system
try:
    from .dynamic_timing_system import DynamicTimingSystem, TimingRegime, get_dynamic_timing_system
    DYNAMIC_TIMING_AVAILABLE = True
except ImportError:
    DYNAMIC_TIMING_AVAILABLE = False
    logger.warning("Dynamic timing system not available")

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Supported data sources."""
    MARKET_DATA = "market_data"
    ORDER_BOOK = "order_book"
    TRADE_HISTORY = "trade_history"
    VOLUME_DATA = "volume_data"
    SENTIMENT_DATA = "sentiment_data"
    ONCHAIN_DATA = "onchain_data"

class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"

@dataclass
class DataPoint:
    """Individual data point with metadata."""
    source: DataSource
    symbol: str
    value: float
    timestamp: float
    quality: DataQuality
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RollingDataSeries:
    """Rolling data series with time-weighted calculations."""
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    time_weights: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: float = field(default_factory=time.time)
    
    # Rolling statistics
    rolling_mean: float = 0.0
    rolling_std: float = 0.0
    rolling_min: float = 0.0
    rolling_max: float = 0.0
    
    # Quality metrics
    quality_score: float = 1.0
    freshness_score: float = 1.0
    completeness_score: float = 1.0

@dataclass
class PullConfig:
    """Configuration for data pulling."""
    base_interval: float
    min_interval: float
    max_interval: float
    quality_threshold: float
    retry_attempts: int
    timeout: float
    batch_size: int

class EnhancedRealTimeDataPuller:
    """
    🚀 Enhanced Real-Time Data Puller for Schwabot Trading
    
    Provides dynamic adaptive data collection with rolling measurements
    and real-time timing triggers for optimal trading performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the enhanced real-time data puller."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.active = False
        self.initialized = False
        
        # Data storage
        self.data_series: Dict[str, RollingDataSeries] = {}
        self.pull_configs: Dict[str, PullConfig] = {}
        
        # Performance tracking
        self.pull_metrics = {
            'total_pulls': 0,
            'successful_pulls': 0,
            'failed_pulls': 0,
            'avg_pull_time': 0.0,
            'last_pull_time': 0.0
        }
        
        # Threading and async
        self.pull_thread = None
        self.stop_event = threading.Event()
        
        # Dynamic timing integration
        if DYNAMIC_TIMING_AVAILABLE:
            self.dynamic_timing = get_dynamic_timing_system()
            self._setup_dynamic_timing_integration()
        else:
            self.dynamic_timing = None
        
        # Callbacks
        self.data_received_callback: Optional[Callable] = None
        self.quality_alert_callback: Optional[Callable] = None
        self.pull_error_callback: Optional[Callable] = None
        
        self._initialize_system()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for data pulling."""
        return {
            'base_pull_interval': 1.0,      # Base 1-second intervals
            'min_pull_interval': 0.1,       # Minimum 100ms intervals
            'max_pull_interval': 10.0,      # Maximum 10-second intervals
            'quality_threshold': 0.7,       # Minimum quality threshold
            'retry_attempts': 3,            # Number of retry attempts
            'pull_timeout': 5.0,            # Pull timeout in seconds
            'batch_size': 100,              # Batch size for processing
            'enable_dynamic_timing': True,  # Enable dynamic timing integration
            'enable_quality_monitoring': True,
            'enable_rolling_metrics': True
        }
    
    def _initialize_system(self) -> None:
        """Initialize the enhanced data pulling system."""
        try:
            self.logger.info("🚀 Initializing Enhanced Real-Time Data Puller...")
            
            # Initialize pull configurations
            self._setup_pull_configurations()
            
            # Initialize data series
            self._initialize_data_series()
            
            # Setup quality monitoring
            if self.config['enable_quality_monitoring']:
                self._setup_quality_monitoring()
            
            self.initialized = True
            self.logger.info("✅ Enhanced Real-Time Data Puller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Enhanced Real-Time Data Puller: {e}")
            self.initialized = False
    
    def _setup_pull_configurations(self) -> None:
        """Setup pull configurations for different data sources."""
        try:
            # Market data configuration
            self.pull_configs[DataSource.MARKET_DATA.value] = PullConfig(
                base_interval=1.0,
                min_interval=0.1,
                max_interval=5.0,
                quality_threshold=0.8,
                retry_attempts=3,
                timeout=2.0,
                batch_size=50
            )
            
            # Order book configuration
            self.pull_configs[DataSource.ORDER_BOOK.value] = PullConfig(
                base_interval=0.5,
                min_interval=0.05,
                max_interval=2.0,
                quality_threshold=0.9,
                retry_attempts=5,
                timeout=1.0,
                batch_size=100
            )
            
            # Trade history configuration
            self.pull_configs[DataSource.TRADE_HISTORY.value] = PullConfig(
                base_interval=2.0,
                min_interval=0.5,
                max_interval=10.0,
                quality_threshold=0.7,
                retry_attempts=2,
                timeout=5.0,
                batch_size=200
            )
            
            # Volume data configuration
            self.pull_configs[DataSource.VOLUME_DATA.value] = PullConfig(
                base_interval=1.0,
                min_interval=0.2,
                max_interval=5.0,
                quality_threshold=0.8,
                retry_attempts=3,
                timeout=3.0,
                batch_size=75
            )
            
        except Exception as e:
            self.logger.error(f"Error setting up pull configurations: {e}")
    
    def _initialize_data_series(self) -> None:
        """Initialize data series for different sources."""
        try:
            for source in DataSource:
                self.data_series[source.value] = RollingDataSeries()
                
        except Exception as e:
            self.logger.error(f"Error initializing data series: {e}")
    
    def _setup_dynamic_timing_integration(self) -> None:
        """Setup integration with dynamic timing system."""
        try:
            if self.dynamic_timing:
                # Set callbacks for dynamic timing
                self.dynamic_timing.set_data_pull_callback(self._adjust_pull_interval)
                
                # Register for regime changes
                self.dynamic_timing.set_regime_change_callback(self._handle_regime_change)
                
                self.logger.info("⚡ Dynamic timing integration configured")
                
        except Exception as e:
            self.logger.error(f"Error setting up dynamic timing integration: {e}")
    
    def _setup_quality_monitoring(self) -> None:
        """Setup quality monitoring system."""
        try:
            self.quality_thresholds = {
                DataQuality.EXCELLENT: 0.9,
                DataQuality.GOOD: 0.7,
                DataQuality.FAIR: 0.5,
                DataQuality.POOR: 0.3,
                DataQuality.UNUSABLE: 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up quality monitoring: {e}")
    
    def start(self) -> bool:
        """Start the enhanced data pulling system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.stop_event.clear()
            
            # Start pull thread
            self.pull_thread = threading.Thread(
                target=self._pull_loop,
                daemon=True,
                name="EnhancedDataPuller"
            )
            self.pull_thread.start()
            
            # Start dynamic timing if available
            if self.dynamic_timing and self.config['enable_dynamic_timing']:
                self.dynamic_timing.start()
            
            self.logger.info("🚀 Enhanced Real-Time Data Puller started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting Enhanced Real-Time Data Puller: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the enhanced data pulling system."""
        try:
            self.active = False
            self.stop_event.set()
            
            # Stop dynamic timing
            if self.dynamic_timing:
                self.dynamic_timing.stop()
            
            # Stop pull thread
            if self.pull_thread and self.pull_thread.is_alive():
                self.pull_thread.join(timeout=5.0)
            
            self.logger.info("🛑 Enhanced Real-Time Data Puller stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Enhanced Real-Time Data Puller: {e}")
            return False
    
    def _pull_loop(self) -> None:
        """Main data pulling loop."""
        try:
            while self.active and not self.stop_event.is_set():
                loop_start = time.time()
                
                # Pull data from all sources
                for source in DataSource:
                    if self._should_pull_data(source):
                        self._pull_data_from_source(source)
                
                # Update rolling metrics
                self._update_rolling_metrics()
                
                # Calculate next pull interval
                next_interval = self._calculate_next_pull_interval()
                
                # Sleep for calculated interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, next_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            self.logger.error(f"❌ Error in pull loop: {e}")
    
    def _should_pull_data(self, source: DataSource) -> bool:
        """Determine if data should be pulled from source."""
        try:
            # Check if source is active
            if not self.active:
                return False
            
            # Check pull configuration
            config = self.pull_configs.get(source.value)
            if not config:
                return False
            
            # Check if enough time has passed since last pull
            last_pull = self.pull_metrics.get('last_pull_time', 0)
            current_time = time.time()
            
            # Calculate adaptive interval
            adaptive_interval = self._calculate_adaptive_interval(source)
            
            return (current_time - last_pull) >= adaptive_interval
            
        except Exception as e:
            self.logger.error(f"Error checking if should pull data: {e}")
            return False
    
    def _pull_data_from_source(self, source: DataSource) -> None:
        """Pull data from a specific source."""
        try:
            start_time = time.time()
            config = self.pull_configs.get(source.value)
            
            if not config:
                return
            
            # Simulate data pulling (replace with actual API calls)
            data_points = self._simulate_data_pull(source, config)
            
            # Process and validate data
            processed_data = self._process_pulled_data(source, data_points)
            
            # Update data series
            self._update_data_series(source, processed_data)
            
            # Update pull metrics
            pull_time = time.time() - start_time
            self.pull_metrics['total_pulls'] += 1
            self.pull_metrics['successful_pulls'] += 1
            self.pull_metrics['avg_pull_time'] = (
                (self.pull_metrics['avg_pull_time'] * 0.9) + (pull_time * 0.1)
            )
            self.pull_metrics['last_pull_time'] = time.time()
            
            # Trigger data received callback
            if self.data_received_callback:
                self.data_received_callback(source, processed_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error pulling data from {source.value}: {e}")
            self.pull_metrics['failed_pulls'] += 1
            
            # Trigger error callback
            if self.pull_error_callback:
                self.pull_error_callback(source, str(e))
    
    def _simulate_data_pull(self, source: DataSource, config: PullConfig) -> List[DataPoint]:
        """Simulate data pulling from source (replace with actual API calls)."""
        try:
            data_points = []
            current_time = time.time()
            
            # Generate simulated data based on source type
            if source == DataSource.MARKET_DATA:
                # Simulate market data
                for i in range(config.batch_size):
                    price = 50000.0 + np.random.normal(0, 100)  # BTC price simulation
                    data_points.append(DataPoint(
                        source=source,
                        symbol="BTC/USD",
                        value=price,
                        timestamp=current_time + i * 0.01,
                        quality=DataQuality.GOOD,
                        metadata={'volume': np.random.uniform(100, 1000)}
                    ))
            
            elif source == DataSource.ORDER_BOOK:
                # Simulate order book data
                for i in range(config.batch_size):
                    bid_price = 50000.0 + np.random.normal(0, 50)
                    ask_price = bid_price + np.random.uniform(1, 10)
                    data_points.append(DataPoint(
                        source=source,
                        symbol="BTC/USD",
                        value=(bid_price + ask_price) / 2,
                        timestamp=current_time + i * 0.005,
                        quality=DataQuality.EXCELLENT,
                        metadata={'bid': bid_price, 'ask': ask_price, 'spread': ask_price - bid_price}
                    ))
            
            elif source == DataSource.TRADE_HISTORY:
                # Simulate trade history
                for i in range(config.batch_size):
                    trade_price = 50000.0 + np.random.normal(0, 200)
                    data_points.append(DataPoint(
                        source=source,
                        symbol="BTC/USD",
                        value=trade_price,
                        timestamp=current_time + i * 0.1,
                        quality=DataQuality.GOOD,
                        metadata={'volume': np.random.uniform(0.1, 10.0)}
                    ))
            
            elif source == DataSource.VOLUME_DATA:
                # Simulate volume data
                for i in range(config.batch_size):
                    volume = np.random.uniform(100, 10000)
                    data_points.append(DataPoint(
                        source=source,
                        symbol="BTC/USD",
                        value=volume,
                        timestamp=current_time + i * 0.02,
                        quality=DataQuality.GOOD,
                        metadata={'price': 50000.0 + np.random.normal(0, 100)}
                    ))
            
            return data_points
            
        except Exception as e:
            self.logger.error(f"Error simulating data pull: {e}")
            return []
    
    def _process_pulled_data(self, source: DataSource, data_points: List[DataPoint]) -> List[DataPoint]:
        """Process and validate pulled data."""
        try:
            processed_data = []
            
            for data_point in data_points:
                # Validate data point
                if self._validate_data_point(data_point):
                    # Assess quality
                    quality = self._assess_data_quality(data_point)
                    data_point.quality = quality
                    
                    # Add to processed data
                    processed_data.append(data_point)
                    
                    # Check quality threshold
                    if quality.value < self.pull_configs[source.value].quality_threshold:
                        if self.quality_alert_callback:
                            self.quality_alert_callback(source, data_point, quality)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing pulled data: {e}")
            return []
    
    def _validate_data_point(self, data_point: DataPoint) -> bool:
        """Validate a data point."""
        try:
            # Check basic requirements
            if not data_point.symbol or not data_point.value or not data_point.timestamp:
                return False
            
            # Check timestamp validity
            current_time = time.time()
            if data_point.timestamp > current_time + 60:  # Future timestamp
                return False
            if data_point.timestamp < current_time - 3600:  # Too old
                return False
            
            # Check value validity
            if not np.isfinite(data_point.value):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data point: {e}")
            return False
    
    def _assess_data_quality(self, data_point: DataPoint) -> DataQuality:
        """Assess the quality of a data point."""
        try:
            quality_score = 1.0
            
            # Check freshness
            current_time = time.time()
            age = current_time - data_point.timestamp
            freshness_score = max(0, 1.0 - (age / 60.0))  # Decay over 1 minute
            quality_score *= freshness_score
            
            # Check value reasonableness
            if data_point.source == DataSource.MARKET_DATA:
                if data_point.value < 1000 or data_point.value > 100000:  # BTC price range
                    quality_score *= 0.5
            
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
                return DataQuality.UNUSABLE
                
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return DataQuality.UNUSABLE
    
    def _update_data_series(self, source: DataSource, data_points: List[DataPoint]) -> None:
        """Update data series with new data points."""
        try:
            series = self.data_series.get(source.value)
            if not series:
                return
            
            # Add data points
            for data_point in data_points:
                series.data_points.append(data_point.value)
                series.time_weights.append(1.0)
            
            # Update rolling statistics
            self._update_rolling_statistics(series)
            
            # Update quality metrics
            self._update_quality_metrics(series, data_points)
            
            series.last_update = time.time()
            
        except Exception as e:
            self.logger.error(f"Error updating data series: {e}")
    
    def _update_rolling_statistics(self, series: RollingDataSeries) -> None:
        """Update rolling statistics for a data series."""
        try:
            if len(series.data_points) < 2:
                return
            
            # Calculate time-weighted statistics
            values = list(series.data_points)
            weights = list(series.time_weights)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                
                # Calculate weighted statistics
                series.rolling_mean = np.average(values, weights=normalized_weights)
                series.rolling_std = np.sqrt(np.average((values - series.rolling_mean) ** 2, weights=normalized_weights))
                series.rolling_min = min(values)
                series.rolling_max = max(values)
            
        except Exception as e:
            self.logger.error(f"Error updating rolling statistics: {e}")
    
    def _update_quality_metrics(self, series: RollingDataSeries, data_points: List[DataPoint]) -> None:
        """Update quality metrics for a data series."""
        try:
            if not data_points:
                return
            
            # Calculate quality scores
            quality_scores = [self.quality_thresholds.get(dp.quality, 0.0) for dp in data_points]
            
            # Update series quality metrics
            series.quality_score = np.mean(quality_scores)
            series.freshness_score = 1.0  # Assuming fresh data
            series.completeness_score = 1.0  # Assuming complete data
            
        except Exception as e:
            self.logger.error(f"Error updating quality metrics: {e}")
    
    def _update_rolling_metrics(self) -> None:
        """Update rolling metrics across all data series."""
        try:
            # Update dynamic timing with volatility and momentum data
            if self.dynamic_timing:
                # Calculate overall volatility
                volatility = self._calculate_overall_volatility()
                self.dynamic_timing.add_volatility_data(volatility)
                
                # Calculate overall momentum
                momentum = self._calculate_overall_momentum()
                self.dynamic_timing.add_momentum_data(momentum)
            
        except Exception as e:
            self.logger.error(f"Error updating rolling metrics: {e}")
    
    def _calculate_overall_volatility(self) -> float:
        """Calculate overall volatility across all data sources."""
        try:
            volatilities = []
            
            for series in self.data_series.values():
                if series.rolling_std > 0:
                    volatilities.append(series.rolling_std / series.rolling_mean)
            
            return np.mean(volatilities) if volatilities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall volatility: {e}")
            return 0.0
    
    def _calculate_overall_momentum(self) -> float:
        """Calculate overall momentum across all data sources."""
        try:
            momentums = []
            
            for series in self.data_series.values():
                if len(series.data_points) >= 2:
                    recent_values = list(series.data_points)[-10:]
                    if len(recent_values) >= 2:
                        momentum = (recent_values[-1] - recent_values[0]) / recent_values[0]
                        momentums.append(momentum)
            
            return np.mean(momentums) if momentums else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall momentum: {e}")
            return 0.0
    
    def _calculate_next_pull_interval(self) -> float:
        """Calculate the next pull interval based on current conditions."""
        try:
            base_interval = self.config['base_pull_interval']
            
            # Apply dynamic timing adjustments if available
            if self.dynamic_timing:
                # Get current regime
                regime = self.dynamic_timing.get_current_regime()
                
                # Apply regime-based adjustments
                regime_multipliers = {
                    TimingRegime.CALM: 1.5,
                    TimingRegime.NORMAL: 1.0,
                    TimingRegime.VOLATILE: 0.5,
                    TimingRegime.EXTREME: 0.2,
                    TimingRegime.CRISIS: 0.1
                }
                
                regime_multiplier = regime_multipliers.get(regime, 1.0)
                base_interval *= regime_multiplier
            
            # Clamp to min/max bounds
            min_interval = self.config['min_pull_interval']
            max_interval = self.config['max_pull_interval']
            
            return max(min_interval, min(max_interval, base_interval))
            
        except Exception as e:
            self.logger.error(f"Error calculating next pull interval: {e}")
            return self.config['base_pull_interval']
    
    def _calculate_adaptive_interval(self, source: DataSource) -> float:
        """Calculate adaptive interval for a specific source."""
        try:
            config = self.pull_configs.get(source.value)
            if not config:
                return self.config['base_pull_interval']
            
            base_interval = config.base_interval
            
            # Apply quality-based adjustments
            series = self.data_series.get(source.value)
            if series and series.quality_score < config.quality_threshold:
                # Increase interval for poor quality data
                base_interval *= 1.5
            
            # Apply dynamic timing adjustments
            if self.dynamic_timing:
                regime = self.dynamic_timing.get_current_regime()
                volatility = self.dynamic_timing._calculate_current_volatility()
                
                # Adjust based on regime and volatility
                if regime == TimingRegime.CRISIS:
                    base_interval *= 0.1
                elif regime == TimingRegime.EXTREME:
                    base_interval *= 0.2
                elif regime == TimingRegime.VOLATILE:
                    base_interval *= 0.5
                elif regime == TimingRegime.CALM:
                    base_interval *= 1.5
            
            # Clamp to source-specific bounds
            return max(config.min_interval, min(config.max_interval, base_interval))
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive interval: {e}")
            return self.config['base_pull_interval']
    
    def _adjust_pull_interval(self, new_interval: float) -> None:
        """Adjust pull interval based on dynamic timing feedback."""
        try:
            # Update base interval
            self.config['base_pull_interval'] = new_interval
            
            self.logger.info(f"⚡ Pull interval adjusted to {new_interval:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error adjusting pull interval: {e}")
    
    def _handle_regime_change(self, old_regime: TimingRegime, new_regime: TimingRegime, 
                            volatility: float, momentum: float) -> None:
        """Handle regime change from dynamic timing system."""
        try:
            self.logger.info(f"🔄 Regime change detected: {old_regime.value} → {new_regime.value}")
            
            # Adjust pull configurations based on new regime
            if new_regime == TimingRegime.CRISIS:
                # Emergency mode - maximum data collection
                for config in self.pull_configs.values():
                    config.base_interval *= 0.1
                    config.quality_threshold *= 0.8
            
            elif new_regime == TimingRegime.EXTREME:
                # High volatility mode
                for config in self.pull_configs.values():
                    config.base_interval *= 0.2
            
            elif new_regime == TimingRegime.VOLATILE:
                # Volatile mode
                for config in self.pull_configs.values():
                    config.base_interval *= 0.5
            
            elif new_regime == TimingRegime.CALM:
                # Calm mode - relaxed collection
                for config in self.pull_configs.values():
                    config.base_interval *= 1.5
            
        except Exception as e:
            self.logger.error(f"Error handling regime change: {e}")
    
    def set_data_received_callback(self, callback: Callable) -> None:
        """Set callback for data received events."""
        self.data_received_callback = callback
    
    def set_quality_alert_callback(self, callback: Callable) -> None:
        """Set callback for quality alert events."""
        self.quality_alert_callback = callback
    
    def set_pull_error_callback(self, callback: Callable) -> None:
        """Set callback for pull error events."""
        self.pull_error_callback = callback
    
    def get_data_series(self, source: DataSource) -> Optional[RollingDataSeries]:
        """Get data series for a specific source."""
        return self.data_series.get(source.value)
    
    def get_pull_metrics(self) -> Dict[str, Any]:
        """Get pull performance metrics."""
        return self.pull_metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'active': self.active,
                'initialized': self.initialized,
                'pull_metrics': self.pull_metrics,
                'data_sources_count': len(self.data_series),
                'dynamic_timing_available': DYNAMIC_TIMING_AVAILABLE,
                'dynamic_timing_active': self.dynamic_timing.active if self.dynamic_timing else False,
                'current_regime': self.dynamic_timing.get_current_regime().value if self.dynamic_timing else 'unknown',
                'data_series_status': {
                    source: {
                        'data_points_count': len(series.data_points),
                        'rolling_mean': series.rolling_mean,
                        'rolling_std': series.rolling_std,
                        'quality_score': series.quality_score,
                        'last_update': series.last_update
                    }
                    for source, series in self.data_series.items()
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

# Global instance
enhanced_data_puller = EnhancedRealTimeDataPuller()

def get_enhanced_data_puller() -> EnhancedRealTimeDataPuller:
    """Get the global EnhancedRealTimeDataPuller instance."""
    return enhanced_data_puller 