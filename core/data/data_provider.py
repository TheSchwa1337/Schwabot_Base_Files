"""
Data Provider Interface
=====================

Defines interfaces for live and historical data providers.
Handles data synchronization and caching.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Base class for data providers"""
    
    @abstractmethod
    def get_price(self, asset: str, timestamp: Optional[datetime] = None) -> float:
        """Get price for an asset at a specific time"""
        pass
        
    @abstractmethod
    def get_volume(self, asset: str, timestamp: Optional[datetime] = None) -> float:
        """Get volume for an asset at a specific time"""
        pass
        
    @abstractmethod
    def get_metrics(self, asset: str, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Get all metrics for an asset at a specific time"""
        pass

class LiveDataProvider(DataProvider):
    """Provides real-time market data"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize live data provider"""
        self.api_key = api_key
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timeout = 60  # seconds
        
    def get_price(self, asset: str, timestamp: Optional[datetime] = None) -> float:
        """Get current price for an asset"""
        if timestamp:
            raise ValueError("Live data provider does not support historical timestamps")
            
        # Check cache
        if self._is_cache_valid(asset, "price"):
            return self.cache[asset]["price"]
            
        # TODO: Implement actual API call
        price = 0.0  # Placeholder
        
        # Update cache
        self._update_cache(asset, "price", price)
        
        return price
        
    def get_volume(self, asset: str, timestamp: Optional[datetime] = None) -> float:
        """Get current volume for an asset"""
        if timestamp:
            raise ValueError("Live data provider does not support historical timestamps")
            
        # Check cache
        if self._is_cache_valid(asset, "volume"):
            return self.cache[asset]["volume"]
            
        # TODO: Implement actual API call
        volume = 0.0  # Placeholder
        
        # Update cache
        self._update_cache(asset, "volume", volume)
        
        return volume
        
    def get_metrics(self, asset: str, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Get all current metrics for an asset"""
        if timestamp:
            raise ValueError("Live data provider does not support historical timestamps")
            
        # Check cache
        if self._is_cache_valid(asset, "metrics"):
            return self.cache[asset]["metrics"]
            
        # Get individual metrics
        price = self.get_price(asset)
        volume = self.get_volume(asset)
        
        # Calculate additional metrics
        metrics = {
            "price": price,
            "volume": volume,
            "volatility": self._calculate_volatility(asset),
            "momentum": self._calculate_momentum(asset),
            "rsi": self._calculate_rsi(asset)
        }
        
        # Update cache
        self._update_cache(asset, "metrics", metrics)
        
        return metrics
        
    def _is_cache_valid(self, asset: str, metric: str) -> bool:
        """Check if cached data is still valid"""
        if asset not in self.cache or metric not in self.cache[asset]:
            return False
            
        cache_time = self.cache[asset]["timestamp"]
        age = (datetime.now() - cache_time).total_seconds()
        return age < self.cache_timeout
        
    def _update_cache(self, asset: str, metric: str, value: Any) -> None:
        """Update cache with new data"""
        if asset not in self.cache:
            self.cache[asset] = {}
            
        self.cache[asset][metric] = value
        self.cache[asset]["timestamp"] = datetime.now()
        
    def _calculate_volatility(self, asset: str) -> float:
        """Calculate price volatility"""
        # TODO: Implement actual calculation
        return 0.0
        
    def _calculate_momentum(self, asset: str) -> float:
        """Calculate price momentum"""
        # TODO: Implement actual calculation
        return 0.0
        
    def _calculate_rsi(self, asset: str) -> float:
        """Calculate RSI indicator"""
        # TODO: Implement actual calculation
        return 0.0

class HistoricalDataProvider(DataProvider):
    """Provides historical market data"""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize historical data provider"""
        self.data_path = data_path or Path("data/historical")
        self.data: Dict[str, pd.DataFrame] = {}
        self._load_data()
        
    def _load_data(self) -> None:
        """Load historical data from files"""
        try:
            for file in self.data_path.glob("*.parquet"):
                asset = file.stem
                self.data[asset] = pd.read_parquet(file)
                logger.info(f"Loaded historical data for {asset}")
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
            
    def get_price(self, asset: str, timestamp: Optional[datetime] = None) -> float:
        """Get historical price for an asset"""
        if asset not in self.data:
            raise ValueError(f"No data available for {asset}")
            
        df = self.data[asset]
        if timestamp:
            # Get closest timestamp
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            return float(df.iloc[idx]["price"])
        else:
            # Get latest price
            return float(df.iloc[-1]["price"])
            
    def get_volume(self, asset: str, timestamp: Optional[datetime] = None) -> float:
        """Get historical volume for an asset"""
        if asset not in self.data:
            raise ValueError(f"No data available for {asset}")
            
        df = self.data[asset]
        if timestamp:
            # Get closest timestamp
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            return float(df.iloc[idx]["volume"])
        else:
            # Get latest volume
            return float(df.iloc[-1]["volume"])
            
    def get_metrics(self, asset: str, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Get all historical metrics for an asset"""
        if asset not in self.data:
            raise ValueError(f"No data available for {asset}")
            
        df = self.data[asset]
        if timestamp:
            # Get closest timestamp
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            return df.iloc[idx].to_dict()
        else:
            # Get latest metrics
            return df.iloc[-1].to_dict()
            
    def get_time_range(self, asset: str) -> Tuple[datetime, datetime]:
        """Get available time range for an asset"""
        if asset not in self.data:
            raise ValueError(f"No data available for {asset}")
            
        df = self.data[asset]
        return df.index[0], df.index[-1]
        
    def get_data_points(self, asset: str, start: Optional[datetime] = None, 
                       end: Optional[datetime] = None) -> pd.DataFrame:
        """Get data points for an asset within a time range"""
        if asset not in self.data:
            raise ValueError(f"No data available for {asset}")
            
        df = self.data[asset]
        if start and end:
            return df[start:end]
        elif start:
            return df[start:]
        elif end:
            return df[:end]
        else:
            return df.copy() 