"""
Market Data Handler for Schwabot System
Handles market data ingestion and processing
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
import ccxt
import numpy as np
from pathlib import Path
from .event_bus import EventBus

@dataclass
class MarketData:
    """Container for market data"""
    timestamp: float
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    depth: Dict[str, List[float]]
    metadata: Dict[str, Any]

class MarketDataHandler:
    """Handles market data ingestion and processing"""
    
    def __init__(self, event_bus: EventBus, exchange_id: str = "coinbase", log_dir: str = "logs"):
        self.event_bus = event_bus
        self.exchange_id = exchange_id
        self.exchange = self._initialize_exchange()
        self.running = False
        self.last_update = 0.0
        self.update_interval = 1.0  # seconds
        
        # Data buffers
        self.price_buffer: List[float] = []
        self.volume_buffer: List[float] = []
        self.depth_buffer: List[Dict] = []
        self.max_buffer_size = 1000
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "market_data.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MarketDataHandler')
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialize exchange connection
        
        Returns:
            CCXT exchange instance
        """
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            self.logger.info(f"Initialized {self.exchange_id} exchange")
            return exchange
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {e}")
            raise
    
    def start(self) -> None:
        """Start market data collection"""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        
        # Start data collection thread
        self._collection_thread = threading.Thread(target=self._collect_data)
        self._collection_thread.daemon = True
        self._collection_thread.start()
        
        self.logger.info("Started market data collection")
    
    def stop(self) -> None:
        """Stop market data collection"""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        self._collection_thread.join()
        
        self.logger.info("Stopped market data collection")
    
    def _collect_data(self) -> None:
        """Collect market data in background thread"""
        while not self._stop_event.is_set():
            try:
                # Get current timestamp
                current_time = datetime.now().timestamp()
                
                # Check update interval
                if current_time - self.last_update < self.update_interval:
                    continue
                
                # Fetch market data
                ticker = self.exchange.fetch_ticker('BTC/USD')
                order_book = self.exchange.fetch_order_book('BTC/USD')
                
                # Process data
                market_data = MarketData(
                    timestamp=current_time,
                    price=ticker['last'],
                    volume=ticker['baseVolume'],
                    bid=ticker['bid'],
                    ask=ticker['ask'],
                    spread=ticker['ask'] - ticker['bid'],
                    depth={
                        'bids': [[price, amount] for price, amount in order_book['bids']],
                        'asks': [[price, amount] for price, amount in order_book['asks']]
                    },
                    metadata={
                        'exchange': self.exchange_id,
                        'symbol': 'BTC/USD',
                        'timestamp': ticker['timestamp']
                    }
                )
                
                # Update buffers
                with self._lock:
                    self.price_buffer.append(market_data.price)
                    self.volume_buffer.append(market_data.volume)
                    self.depth_buffer.append(market_data.depth)
                    
                    # Trim buffers if needed
                    if len(self.price_buffer) > self.max_buffer_size:
                        self.price_buffer.pop(0)
                        self.volume_buffer.pop(0)
                        self.depth_buffer.pop(0)
                
                # Calculate metrics
                self._calculate_metrics(market_data)
                
                # Update last update time
                self.last_update = current_time
                
            except Exception as e:
                self.logger.error(f"Error collecting market data: {e}")
                continue
    
    def _calculate_metrics(self, data: MarketData) -> None:
        """
        Calculate market metrics and update event bus
        
        Args:
            data: Market data
        """
        try:
            # Calculate price metrics
            price_std = np.std(self.price_buffer) if self.price_buffer else 0.0
            price_mean = np.mean(self.price_buffer) if self.price_buffer else 0.0
            price_volatility = price_std / price_mean if price_mean > 0 else 0.0
            
            # Calculate volume metrics
            volume_std = np.std(self.volume_buffer) if self.volume_buffer else 0.0
            volume_mean = np.mean(self.volume_buffer) if self.volume_buffer else 0.0
            volume_volatility = volume_std / volume_mean if volume_mean > 0 else 0.0
            
            # Calculate depth metrics
            bid_depth = sum(amount for _, amount in data.depth['bids'])
            ask_depth = sum(amount for _, amount in data.depth['asks'])
            depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 0.0
            
            # Update event bus
            self.event_bus.update("price", data.price, "market_data", {
                "volatility": price_volatility,
                "mean": price_mean,
                "std": price_std
            })
            
            self.event_bus.update("volume", data.volume, "market_data", {
                "volatility": volume_volatility,
                "mean": volume_mean,
                "std": volume_std
            })
            
            self.event_bus.update("spread", data.spread, "market_data", {
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "depth_ratio": depth_ratio
            })
            
            # Calculate velocity class
            velocity = abs(data.price - price_mean) / price_std if price_std > 0 else 0.0
            velocity_class = "HIGH" if velocity > 2.0 else "MEDIUM" if velocity > 1.0 else "LOW"
            self.event_bus.update("velocity_class", velocity_class, "market_data", {
                "velocity": velocity,
                "price_std": price_std
            })
            
            # Calculate liquidity status
            liquidity_status = "vacuum" if depth_ratio < 0.5 or depth_ratio > 2.0 else "normal"
            self.event_bus.update("liquidity_status", liquidity_status, "market_data", {
                "depth_ratio": depth_ratio,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
    
    def get_market_data(self) -> Optional[MarketData]:
        """
        Get latest market data
        
        Returns:
            Latest MarketData object or None
        """
        with self._lock:
            if not self.price_buffer:
                return None
            
            return MarketData(
                timestamp=self.last_update,
                price=self.price_buffer[-1],
                volume=self.volume_buffer[-1],
                bid=0.0,  # These would need to be stored separately
                ask=0.0,  # if needed for historical data
                spread=0.0,
                depth=self.depth_buffer[-1],
                metadata={
                    "exchange": self.exchange_id,
                    "symbol": "BTC/USD"
                }
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current market metrics
        
        Returns:
            Dictionary of current metrics
        """
        with self._lock:
            return {
                "price": {
                    "current": self.price_buffer[-1] if self.price_buffer else 0.0,
                    "mean": np.mean(self.price_buffer) if self.price_buffer else 0.0,
                    "std": np.std(self.price_buffer) if self.price_buffer else 0.0
                },
                "volume": {
                    "current": self.volume_buffer[-1] if self.volume_buffer else 0.0,
                    "mean": np.mean(self.volume_buffer) if self.volume_buffer else 0.0,
                    "std": np.std(self.volume_buffer) if self.volume_buffer else 0.0
                },
                "last_update": self.last_update
            } 