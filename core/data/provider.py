from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Abstract interface for data providers."""
    @abstractmethod
    def get_price(self, symbol: str, timestamp: Optional[int] = None) -> Optional[float]:
        pass
    @abstractmethod
    def get_ohlcv(self, symbol: str, start: int, end: int, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        pass

class BacktestDataProvider(DataProvider):
    """Historical data provider for backtesting."""
    def __init__(self, data_path: Path):
        if not data_path.exists():
            raise FileNotFoundError(f"Backtest data file not found: {data_path}")
        try:
            self.data = pd.read_parquet(data_path)
            if 'timestamp' in self.data.columns and not isinstance(self.data.index, pd.DatetimeIndex):
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='s')
                self.data = self.data.set_index('timestamp').sort_index()
        except Exception as e:
            logger.error(f"Failed to load backtest data from {data_path}: {e}")
            raise
    def get_price(self, symbol: str, timestamp: Optional[int] = None) -> Optional[float]:
        try:
            if self.data.empty:
                return None
            if timestamp is None:
                return self.data['close'].iloc[-1]
            lookup_time = pd.to_datetime(timestamp, unit='s')
            if lookup_time in self.data.index:
                return self.data.loc[lookup_time, 'close']
            idx = self.data.index.get_loc(lookup_time, method='pad')
            if idx >= 0:
                return self.data.iloc[idx]['close']
            return None
        except Exception as e:
            logger.warning(f"Error getting price: {e}")
            return None
    def get_ohlcv(self, symbol: str, start: int, end: int, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        try:
            start_time = pd.to_datetime(start, unit='s')
            end_time = pd.to_datetime(end, unit='s')
            ohlcv_data = self.data.loc[start_time:end_time].copy()
            if ohlcv_data.empty:
                return None
            if not ohlcv_data.index.name == 'timestamp':
                ohlcv_data.reset_index(inplace=True)
                ohlcv_data.rename(columns={ohlcv_data.index.name: 'timestamp'}, inplace=True)
            return ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.warning(f"Error getting OHLCV: {e}")
            return None

class LiveDataProvider(DataProvider):
    """Live API data provider (to be implemented)."""
    def __init__(self, api_key: str, api_secret: str, exchange_id: str = 'binance'):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_id = exchange_id
        logger.info(f"LiveDataProvider initialized for exchange: {exchange_id} (API client stub)")
    def get_price(self, symbol: str, timestamp: Optional[int] = None) -> Optional[float]:
        logger.warning(f"Live price fetching for {symbol} not yet implemented (stub).")
        return None
    def get_ohlcv(self, symbol: str, start: int, end: int, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        logger.warning(f"Live OHLCV fetching for {symbol} not yet implemented (stub).")
        return None 