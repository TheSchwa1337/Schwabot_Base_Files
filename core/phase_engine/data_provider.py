"""
Data Provider Module
===================

Provides abstract and concrete implementations for data providers
that can be used in both live and historical contexts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import requests
from datetime import datetime

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def get_price(self, pair: str = 'BTC-USD', timestamp: Optional[datetime] = None) -> float:
        """Get price for a trading pair at optional timestamp"""
        pass

class HistoricalDataProvider(DataProvider):
    """Provider for historical price data"""
    
    def __init__(self, historical_data: Dict[str, Dict[datetime, float]]):
        self.historical_data = historical_data
        
    def get_price(self, pair: str = 'BTC-USD', timestamp: Optional[datetime] = None) -> float:
        if pair not in self.historical_data:
            raise ValueError(f"No historical data for pair {pair}")
            
        if timestamp is None:
            # Return most recent price
            return max(self.historical_data[pair].items(), key=lambda x: x[0])[1]
            
        # Find closest timestamp
        timestamps = sorted(self.historical_data[pair].keys())
        closest = min(timestamps, key=lambda x: abs((x - timestamp).total_seconds()))
        return self.historical_data[pair][closest]

class CoinbaseDataProvider(DataProvider):
    """Provider for live Coinbase price data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.coinbase.com/v2"
        
    def get_price(self, pair: str = 'BTC-USD', timestamp: Optional[datetime] = None) -> float:
        if timestamp is not None:
            # For historical data, use Coinbase's historical endpoint
            endpoint = f"{self.base_url}/prices/{pair}/spot"
            params = {'date': timestamp.strftime('%Y-%m-%d')}
        else:
            # For current price
            endpoint = f"{self.base_url}/prices/{pair}/spot"
            params = {}
            
        headers = {}
        if self.api_key:
            headers['CB-ACCESS-KEY'] = self.api_key
            
        response = requests.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        
        return float(response.json()['data']['amount']) 