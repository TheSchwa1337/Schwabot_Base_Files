#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Time Market Data Module
=============================
Provides real-time market data functionality for the Schwabot trading system.

Mathematical Core:
M(t) = {P_t, V_t, O_t, H_t, L_t, C_t} => Streaming Data Vector
- P_t: price at time t
- V_t: volume at t
- ΔP/Δt: velocity for signal trigger

This module implements the foundation data layer that feeds into:
- order_book_analyzer.py
- trading_strategy_executor.py
- strategy_router.py
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import websockets
import json

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
    from core.unified_mathematical_bridge import UnifiedMathematicalBridge
    from core.unified_mathematical_integration_methods import UnifiedMathematicalIntegrationMethods
    from core.unified_mathematical_performance_monitor import UnifiedMathematicalPerformanceMonitor
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Mathematical infrastructure not available - using fallback")


class DataStreamType(Enum):
    """Market data stream types."""
    OHLCV = "ohlcv"
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    DEPTH = "depth"


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class MarketDataPoint:
    """Single market data point with mathematical properties."""
    timestamp: float
    price: float
    volume: float
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    
    # Mathematical properties
    price_velocity: float = 0.0
    volume_momentum: float = 0.0
    volatility: float = 0.0
    mathematical_signature: str = ""


@dataclass
class MarketDataStream:
    """Real-time market data stream with mathematical analysis."""
    symbol: str
    data_type: DataStreamType
    data_points: List[MarketDataPoint] = field(default_factory=list)
    mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
    regime_classification: MarketRegime = MarketRegime.CALM
    last_update: float = field(default_factory=time.time)


@dataclass
class RealTimeMarketConfig:
    """Configuration for real-time market data system."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    update_frequency: float = 1.0  # seconds
    max_data_points: int = 1000
    mathematical_analysis_enabled: bool = True
    websocket_urls: Dict[str, str] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)


class RealTimeMarketData:
    """
    Real-Time Market Data System
    
    Implements the mathematical foundation layer:
    M(t) = {P_t, V_t, O_t, H_t, L_t, C_t} => Streaming Data Vector
    
    Provides real-time market data with mathematical analysis and
    feeds into the trading pipeline.
    """
    
    def __init__(self, config: Optional[RealTimeMarketConfig] = None):
        """Initialize the real-time market data system."""
        self.config = config or RealTimeMarketConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data streams
        self.market_streams: Dict[str, MarketDataStream] = {}
        self.active_streams: Dict[str, bool] = {}
        
        # Mathematical infrastructure
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_bridge = UnifiedMathematicalBridge()
            self.math_integration = UnifiedMathematicalIntegrationMethods()
            self.math_monitor = UnifiedMathematicalPerformanceMonitor()
        else:
            self.math_bridge = None
            self.math_integration = None
            self.math_monitor = None
        
        # Performance tracking
        self.performance_metrics = {
            'data_points_processed': 0,
            'mathematical_analyses': 0,
            'regime_classifications': 0,
            'average_processing_time': 0.0
        }
        
        # System state
        self.initialized = False
        self.active = False
        self.websocket_connections: Dict[str, Any] = {}
        
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the market data system."""
        try:
            self.logger.info("Initializing Real-Time Market Data System")
            
            # Set up default websocket URLs
            if not self.config.websocket_urls:
                self.config.websocket_urls = {
                    'binance': 'wss://stream.binance.com:9443/ws/',
                    'coinbase': 'wss://ws-feed.pro.coinbase.com',
                    'kraken': 'wss://ws.kraken.com'
                }
            
            self.initialized = True
            self.logger.info("✅ Real-Time Market Data System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Real-Time Market Data System: {e}")
            self.initialized = False
    
    async def start_data_stream(self, symbol: str, data_type: DataStreamType = DataStreamType.TICKER) -> bool:
        """Start a real-time data stream for a symbol."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.logger.info(f"Starting data stream for {symbol} ({data_type.value})")
            
            # Create market stream
            stream = MarketDataStream(
                symbol=symbol,
                data_type=data_type
            )
            self.market_streams[symbol] = stream
            self.active_streams[symbol] = True
            
            # Start websocket connection
            await self._connect_websocket(symbol, data_type)
            
            self.logger.info(f"✅ Data stream started for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting data stream for {symbol}: {e}")
            return False
    
    async def _connect_websocket(self, symbol: str, data_type: DataStreamType) -> None:
        """Connect to websocket for real-time data."""
        try:
            # Use Binance as default
            base_url = self.config.websocket_urls.get('binance', 'wss://stream.binance.com:9443/ws/')
            
            # Create subscription message
            if data_type == DataStreamType.TICKER:
                stream_name = f"{symbol.lower()}@ticker"
            elif data_type == DataStreamType.OHLCV:
                stream_name = f"{symbol.lower()}@kline_1m"
            elif data_type == DataStreamType.ORDERBOOK:
                stream_name = f"{symbol.lower()}@depth"
            else:
                stream_name = f"{symbol.lower()}@trade"
            
            url = f"{base_url}{stream_name}"
            
            # Start websocket connection
            self.websocket_connections[symbol] = await websockets.connect(url)
            
            # Start data processing task
            asyncio.create_task(self._process_websocket_data(symbol))
            
        except Exception as e:
            self.logger.error(f"❌ Error connecting websocket for {symbol}: {e}")
    
    async def _process_websocket_data(self, symbol: str) -> None:
        """Process incoming websocket data."""
        try:
            websocket = self.websocket_connections.get(symbol)
            if not websocket:
                return
            
            async for message in websocket:
                if not self.active_streams.get(symbol, False):
                    break
                
                # Parse message
                data = json.loads(message)
                
                # Create market data point
                data_point = self._parse_market_data(data, symbol)
                
                # Add to stream
                stream = self.market_streams.get(symbol)
                if stream:
                    stream.data_points.append(data_point)
                    
                    # Maintain max data points
                    if len(stream.data_points) > self.config.max_data_points:
                        stream.data_points.pop(0)
                    
                    # Update timestamp
                    stream.last_update = time.time()
                    
                    # Perform mathematical analysis
                    if self.config.mathematical_analysis_enabled:
                        await self._perform_mathematical_analysis(symbol, data_point)
                    
                    # Update performance metrics
                    self.performance_metrics['data_points_processed'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ Error processing websocket data for {symbol}: {e}")
    
    def _parse_market_data(self, data: Dict[str, Any], symbol: str) -> MarketDataPoint:
        """Parse raw market data into structured format."""
        try:
            timestamp = time.time()
            
            # Extract price and volume
            if 'p' in data:  # Binance ticker format
                price = float(data['p'])
                volume = float(data['v'])
            elif 'price' in data:  # Generic format
                price = float(data['price'])
                volume = float(data.get('volume', 0))
            else:
                price = 0.0
                volume = 0.0
            
            # Calculate mathematical properties
            price_velocity = self._calculate_price_velocity(symbol, price)
            volume_momentum = self._calculate_volume_momentum(symbol, volume)
            volatility = self._calculate_volatility(symbol, price)
            
            # Create mathematical signature
            mathematical_signature = self._create_mathematical_signature(
                price, volume, price_velocity, volume_momentum, volatility
            )
            
            return MarketDataPoint(
                timestamp=timestamp,
                price=price,
                volume=volume,
                price_velocity=price_velocity,
                volume_momentum=volume_momentum,
                volatility=volatility,
                mathematical_signature=mathematical_signature
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing market data: {e}")
            return MarketDataPoint(timestamp=time.time(), price=0.0, volume=0.0)
    
    def _calculate_price_velocity(self, symbol: str, current_price: float) -> float:
        """Calculate price velocity (ΔP/Δt)."""
        try:
            stream = self.market_streams.get(symbol)
            if not stream or len(stream.data_points) < 2:
                return 0.0
            
            # Get previous price
            prev_point = stream.data_points[-2]
            prev_price = prev_point.price
            
            # Calculate velocity
            time_diff = time.time() - prev_point.timestamp
            if time_diff > 0:
                velocity = (current_price - prev_price) / time_diff
                return velocity
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating price velocity: {e}")
            return 0.0
    
    def _calculate_volume_momentum(self, symbol: str, current_volume: float) -> float:
        """Calculate volume momentum."""
        try:
            stream = self.market_streams.get(symbol)
            if not stream or len(stream.data_points) < 2:
                return 0.0
            
            # Get previous volume
            prev_point = stream.data_points[-2]
            prev_volume = prev_point.volume
            
            # Calculate momentum
            momentum = current_volume - prev_volume
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volume momentum: {e}")
            return 0.0
    
    def _calculate_volatility(self, symbol: str, current_price: float) -> float:
        """Calculate price volatility."""
        try:
            stream = self.market_streams.get(symbol)
            if not stream or len(stream.data_points) < 10:
                return 0.0
            
            # Get recent prices
            recent_prices = [point.price for point in stream.data_points[-10:]]
            recent_prices.append(current_price)
            
            # Calculate standard deviation
            prices_array = np.array(recent_prices)
            volatility = np.std(prices_array)
            return float(volatility)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volatility: {e}")
            return 0.0
    
    def _create_mathematical_signature(self, price: float, volume: float, 
                                     velocity: float, momentum: float, volatility: float) -> str:
        """Create mathematical signature for data point."""
        try:
            # Combine mathematical properties into signature
            signature_components = [
                f"P:{price:.6f}",
                f"V:{volume:.6f}",
                f"ΔP:{velocity:.6f}",
                f"ΔV:{momentum:.6f}",
                f"σ:{volatility:.6f}"
            ]
            return "|".join(signature_components)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating mathematical signature: {e}")
            return ""
    
    async def _perform_mathematical_analysis(self, symbol: str, data_point: MarketDataPoint) -> None:
        """Perform mathematical analysis on market data."""
        try:
            if not self.math_bridge:
                return
            
            # Prepare market data for mathematical analysis
            market_data = {
                'symbol': symbol,
                'price': data_point.price,
                'volume': data_point.volume,
                'price_velocity': data_point.price_velocity,
                'volume_momentum': data_point.volume_momentum,
                'volatility': data_point.volatility,
                'mathematical_signature': data_point.mathematical_signature,
                'timestamp': data_point.timestamp
            }
            
            # Perform mathematical integration
            result = self.math_bridge.integrate_all_mathematical_systems(
                market_data, {}
            )
            
            # Update stream with mathematical analysis
            stream = self.market_streams.get(symbol)
            if stream:
                stream.mathematical_analysis = {
                    'confidence': result.overall_confidence,
                    'connections': len(result.connections),
                    'performance_metrics': result.performance_metrics,
                    'mathematical_signature': result.mathematical_signature
                }
                
                # Classify market regime
                stream.regime_classification = self._classify_market_regime(
                    data_point, stream.mathematical_analysis
                )
            
            # Update performance metrics
            self.performance_metrics['mathematical_analyses'] += 1
            self.performance_metrics['regime_classifications'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error performing mathematical analysis: {e}")
    
    def _classify_market_regime(self, data_point: MarketDataPoint, 
                              mathematical_analysis: Dict[str, Any]) -> MarketRegime:
        """Classify market regime based on mathematical analysis."""
        try:
            # Use volatility and mathematical confidence for classification
            volatility = data_point.volatility
            confidence = mathematical_analysis.get('confidence', 0.5)
            
            if volatility > 0.05:  # High volatility
                return MarketRegime.VOLATILE
            elif volatility < 0.01:  # Low volatility
                return MarketRegime.CALM
            elif data_point.price_velocity > 0.001:  # Trending up
                return MarketRegime.TRENDING_UP
            elif data_point.price_velocity < -0.001:  # Trending down
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"❌ Error classifying market regime: {e}")
            return MarketRegime.CALM
    
    def get_market_data(self, symbol: str) -> Optional[MarketDataStream]:
        """Get current market data for a symbol."""
        return self.market_streams.get(symbol)
    
    def get_all_market_data(self) -> Dict[str, MarketDataStream]:
        """Get all market data streams."""
        return self.market_streams.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return self.performance_metrics.copy()
    
    def stop_data_stream(self, symbol: str) -> bool:
        """Stop a data stream."""
        try:
            self.active_streams[symbol] = False
            
            # Close websocket connection
            websocket = self.websocket_connections.get(symbol)
            if websocket:
                asyncio.create_task(websocket.close())
                del self.websocket_connections[symbol]
            
            self.logger.info(f"✅ Data stream stopped for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping data stream for {symbol}: {e}")
            return False
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info("✅ Real-Time Market Data System activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating Real-Time Market Data System: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            
            # Stop all streams
            for symbol in list(self.active_streams.keys()):
                self.stop_data_stream(symbol)
            
            self.logger.info("✅ Real-Time Market Data System deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating Real-Time Market Data System: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'active_streams': len([s for s in self.active_streams.values() if s]),
            'total_streams': len(self.market_streams),
            'performance_metrics': self.performance_metrics,
            'config': {
                'enabled': self.config.enabled,
                'update_frequency': self.config.update_frequency,
                'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled
            }
        }


def create_real_time_market_data(config: Optional[RealTimeMarketConfig] = None) -> RealTimeMarketData:
    """Factory function to create RealTimeMarketData instance."""
    return RealTimeMarketData(config)


async def main():
    """Main function for testing."""
    # Create configuration
    config = RealTimeMarketConfig(
        enabled=True,
        debug=True,
        update_frequency=1.0,
        mathematical_analysis_enabled=True
    )
    
    # Create market data system
    market_data = create_real_time_market_data(config)
    
    # Activate system
    market_data.activate()
    
    # Start data streams
    await market_data.start_data_stream("BTCUSDT", DataStreamType.TICKER)
    await market_data.start_data_stream("ETHUSDT", DataStreamType.TICKER)
    
    # Run for some time
    await asyncio.sleep(30)
    
    # Get status
    status = market_data.get_status()
    print(f"System Status: {status}")
    
    # Get market data
    btc_data = market_data.get_market_data("BTCUSDT")
    if btc_data:
        print(f"BTC Data Points: {len(btc_data.data_points)}")
        if btc_data.data_points:
            latest = btc_data.data_points[-1]
            print(f"Latest BTC Price: ${latest.price}")
            print(f"Regime: {btc_data.regime_classification.value}")
    
    # Deactivate system
    market_data.deactivate()


if __name__ == "__main__":
    asyncio.run(main())
