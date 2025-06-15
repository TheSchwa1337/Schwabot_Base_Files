"""
Data Provider Interface
======================

Provides a standardized interface for market data access, supporting both
live trading via CCXT and backtesting with historical data. Designed for
BTC/USDC trading with comprehensive error handling and validation.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Import configuration management
from config import load_config, ensure_config_exists, ConfigError

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Standardized market data container"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    high_24h: float
    low_24h: float
    change_24h: float
    metadata: Dict[str, Any]

@dataclass
class OrderBookData:
    """Order book data container"""
    symbol: str
    timestamp: datetime
    bids: List[List[float]]  # [[price, volume], ...]
    asks: List[List[float]]  # [[price, volume], ...]
    spread: float
    depth: Dict[str, float]  # {'bid_depth': float, 'ask_depth': float}

@dataclass
class TradeData:
    """Individual trade data container"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: str

class DataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get comprehensive market data for a symbol"""
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBookData:
        """Get order book data for a symbol"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                since: datetime, limit: int = 1000) -> pd.DataFrame:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the data provider is connected"""
        pass

class CCXTDataProvider(DataProvider):
    """CCXT-based data provider for live trading"""
    
    def __init__(self, exchange_id: str = 'binance', config_file: str = 'trading_config.yaml'):
        self.exchange_id = exchange_id
        self.exchange = None
        self.config = self._load_config(config_file)
        self.supported_symbols = self.config.get('supported_symbols', ['BTC/USDT', 'BTC/USDC'])
        self.rate_limit_delay = self.config.get('rate_limit_delay', 0.1)
        
        # Connection state
        self._connected = False
        self._last_request_time = 0.0
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load trading configuration with defaults"""
        default_config = {
            'supported_symbols': ['BTC/USDT', 'BTC/USDC', 'ETH/USDT', 'ETH/USDC'],
            'rate_limit_delay': 0.1,
            'sandbox_mode': True,
            'api_credentials': {
                'api_key': '',
                'secret': '',
                'password': '',  # For some exchanges
                'sandbox': True
            },
            'request_timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 1.0
        }
        
        try:
            config_path = ensure_config_exists(config_file, default_config)
            return load_config(config_file)
        except ConfigError as e:
            logger.warning(f"Config error: {e}. Using defaults.")
            return default_config
    
    async def initialize(self) -> None:
        """Initialize the CCXT exchange connection"""
        try:
            import ccxt.async_support as ccxt
            
            # Get exchange class
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Configure exchange
            config = {
                'enableRateLimit': True,
                'timeout': self.config.get('request_timeout', 30) * 1000,
                'sandbox': self.config.get('sandbox_mode', True)
            }
            
            # Add API credentials if provided
            api_creds = self.config.get('api_credentials', {})
            if api_creds.get('api_key'):
                config.update({
                    'apiKey': api_creds['api_key'],
                    'secret': api_creds['secret'],
                    'password': api_creds.get('password', ''),
                    'sandbox': api_creds.get('sandbox', True)
                })
            
            self.exchange = exchange_class(config)
            
            # Test connection
            await self.exchange.load_markets()
            self._connected = True
            
            logger.info(f"CCXT {self.exchange_id} provider initialized successfully")
            
        except ImportError:
            raise ConfigError("CCXT library not installed. Run: pip install ccxt")
        except Exception as e:
            logger.error(f"Failed to initialize CCXT provider: {e}")
            raise ConfigError(f"CCXT initialization failed: {e}")
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if not self._connected:
            await self.initialize()
        
        await self._rate_limit()
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get comprehensive market data for a symbol"""
        if not self._connected:
            await self.initialize()
        
        await self._rate_limit()
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                price=float(ticker['last']),
                volume=float(ticker['baseVolume'] or 0),
                bid=float(ticker['bid'] or 0),
                ask=float(ticker['ask'] or 0),
                spread=float((ticker['ask'] or 0) - (ticker['bid'] or 0)),
                high_24h=float(ticker['high'] or 0),
                low_24h=float(ticker['low'] or 0),
                change_24h=float(ticker['change'] or 0),
                metadata={
                    'exchange': self.exchange_id,
                    'quote_volume': ticker.get('quoteVolume', 0),
                    'vwap': ticker.get('vwap', 0),
                    'open': ticker.get('open', 0),
                    'close': ticker.get('close', 0)
                }
            )
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            raise
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBookData:
        """Get order book data for a symbol"""
        if not self._connected:
            await self.initialize()
        
        await self._rate_limit()
        
        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            
            bids = [[float(price), float(volume)] for price, volume in order_book['bids']]
            asks = [[float(price), float(volume)] for price, volume in order_book['asks']]
            
            # Calculate depth
            bid_depth = sum(volume for _, volume in bids[:10])  # Top 10 levels
            ask_depth = sum(volume for _, volume in asks[:10])
            
            spread = asks[0][0] - bids[0][0] if bids and asks else 0.0
            
            return OrderBookData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(order_book['timestamp'] / 1000),
                bids=bids,
                asks=asks,
                spread=spread,
                depth={'bid_depth': bid_depth, 'ask_depth': ask_depth}
            )
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                since: datetime, limit: int = 1000) -> pd.DataFrame:
        """Get historical OHLCV data"""
        if not self._connected:
            await self.initialize()
        
        await self._rate_limit()
        
        try:
            since_timestamp = int(since.timestamp() * 1000)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since_timestamp, limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    async def is_connected(self) -> bool:
        """Check if the data provider is connected"""
        return self._connected and self.exchange is not None
    
    async def close(self) -> None:
        """Close the exchange connection"""
        if self.exchange:
            await self.exchange.close()
            self._connected = False
            logger.info(f"CCXT {self.exchange_id} provider closed")

class BacktestDataProvider(DataProvider):
    """Data provider for backtesting using historical data"""
    
    def __init__(self, data_directory: str = 'data/historical'):
        self.data_directory = Path(data_directory)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.current_time = datetime.now()
        self.simulation_speed = 1.0  # 1.0 = real-time, higher = faster
        
    def set_simulation_time(self, timestamp: datetime) -> None:
        """Set the current simulation time for backtesting"""
        self.current_time = timestamp
    
    def load_historical_data(self, symbol: str, filename: str) -> None:
        """Load historical data from file"""
        file_path = self.data_directory / filename
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.data_cache[symbol] = df
            logger.info(f"Loaded historical data for {symbol}: {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            raise
    
    async def get_price(self, symbol: str) -> float:
        """Get price at current simulation time"""
        if symbol not in self.data_cache:
            raise ValueError(f"No historical data loaded for {symbol}")
        
        df = self.data_cache[symbol]
        
        # Find closest timestamp
        closest_idx = df.index.get_indexer([self.current_time], method='nearest')[0]
        if closest_idx == -1:
            raise ValueError(f"No data available for {symbol} at {self.current_time}")
        
        return float(df.iloc[closest_idx]['close'])
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data at current simulation time"""
        if symbol not in self.data_cache:
            raise ValueError(f"No historical data loaded for {symbol}")
        
        df = self.data_cache[symbol]
        
        # Find closest timestamp
        closest_idx = df.index.get_indexer([self.current_time], method='nearest')[0]
        if closest_idx == -1:
            raise ValueError(f"No data available for {symbol} at {self.current_time}")
        
        row = df.iloc[closest_idx]
        
        # Calculate 24h change if we have enough data
        change_24h = 0.0
        if closest_idx >= 24:  # Assuming hourly data
            prev_price = df.iloc[closest_idx - 24]['close']
            change_24h = row['close'] - prev_price
        
        return MarketData(
            symbol=symbol,
            timestamp=row.name,
            price=float(row['close']),
            volume=float(row['volume']),
            bid=float(row['close'] * 0.999),  # Simulate bid/ask spread
            ask=float(row['close'] * 1.001),
            spread=float(row['close'] * 0.002),
            high_24h=float(row['high']),
            low_24h=float(row['low']),
            change_24h=change_24h,
            metadata={
                'source': 'backtest',
                'open': float(row['open']),
                'simulation_time': self.current_time.isoformat()
            }
        )
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBookData:
        """Simulate order book data for backtesting"""
        market_data = await self.get_market_data(symbol)
        
        # Generate synthetic order book around current price
        price = market_data.price
        spread_pct = 0.001  # 0.1% spread
        
        bids = []
        asks = []
        
        for i in range(limit // 2):
            bid_price = price * (1 - spread_pct * (i + 1))
            ask_price = price * (1 + spread_pct * (i + 1))
            
            # Simulate decreasing volume with distance from mid price
            volume = market_data.volume * 0.01 / (i + 1)
            
            bids.append([bid_price, volume])
            asks.append([ask_price, volume])
        
        return OrderBookData(
            symbol=symbol,
            timestamp=self.current_time,
            bids=bids,
            asks=asks,
            spread=price * spread_pct * 2,
            depth={'bid_depth': sum(v for _, v in bids[:10]), 
                  'ask_depth': sum(v for _, v in asks[:10])}
        )
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                since: datetime, limit: int = 1000) -> pd.DataFrame:
        """Get historical data from cache"""
        if symbol not in self.data_cache:
            raise ValueError(f"No historical data loaded for {symbol}")
        
        df = self.data_cache[symbol]
        
        # Filter by date range
        mask = (df.index >= since) & (df.index <= self.current_time)
        filtered_df = df.loc[mask].tail(limit)
        
        return filtered_df
    
    async def is_connected(self) -> bool:
        """Backtest provider is always 'connected'"""
        return True

class DataProviderFactory:
    """Factory for creating data providers"""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> DataProvider:
        """
        Create a data provider instance
        
        Args:
            provider_type: 'ccxt' for live trading, 'backtest' for historical data
            **kwargs: Provider-specific arguments
            
        Returns:
            DataProvider instance
        """
        if provider_type.lower() == 'ccxt':
            return CCXTDataProvider(**kwargs)
        elif provider_type.lower() == 'backtest':
            return BacktestDataProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

# Export commonly used items
__all__ = [
    'MarketData',
    'OrderBookData', 
    'TradeData',
    'DataProvider',
    'CCXTDataProvider',
    'BacktestDataProvider',
    'DataProviderFactory'
] 