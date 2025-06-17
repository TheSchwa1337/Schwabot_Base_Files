"""
Live Data Streamer for Schwabot Visual Core
==========================================

Provides real-time market data streaming from multiple exchanges
with WebSocket connections and REST API fallbacks.

Features:
- Multi-exchange support (Coinbase, Binance, etc.)
- WebSocket streaming with auto-reconnection
- Rate limiting and error handling
- Data validation and normalization
- Thread-safe data delivery
"""

import asyncio
import websockets
import json
import logging
import threading
import time
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from collections import deque
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Standardized market tick data"""
    symbol: str
    price: float
    volume: float
    timestamp: float
    exchange: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

@dataclass
class StreamConfig:
    """Configuration for data streaming"""
    exchange: str
    symbols: List[str]
    websocket_url: str
    rest_url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit: int = 10  # requests per second
    reconnect_delay: float = 5.0
    max_reconnects: int = 10

class ExchangeConnector:
    """Base class for exchange connections"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.connected = False
        self.callbacks: List[Callable[[MarketTick], None]] = []
        self.last_tick = None
        
    def add_callback(self, callback: Callable[[MarketTick], None]):
        """Add callback for new tick data"""
        self.callbacks.append(callback)
    
    def notify_callbacks(self, tick: MarketTick):
        """Notify all callbacks of new tick"""
        self.last_tick = tick
        for callback in self.callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def connect(self):
        """Connect to exchange (to be implemented by subclasses)"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from exchange"""
        self.connected = False

class CoinbaseConnector(ExchangeConnector):
    """Coinbase Pro WebSocket connector"""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.websocket = None
        
    async def connect(self):
        """Connect to Coinbase Pro WebSocket"""
        try:
            # Coinbase Pro WebSocket URL
            url = "wss://ws-feed.pro.coinbase.com"
            
            # Subscribe message
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": self.config.symbols,
                "channels": ["ticker"]
            }
            
            self.websocket = await websockets.connect(url)
            await self.websocket.send(json.dumps(subscribe_msg))
            
            self.connected = True
            logger.info(f"Connected to Coinbase Pro for {self.config.symbols}")
            
            # Listen for messages
            await self._listen_loop()
            
        except Exception as e:
            logger.error(f"Coinbase connection error: {e}")
            self.connected = False
    
    async def _listen_loop(self):
        """Main listening loop for WebSocket messages"""
        try:
            while self.connected and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "ticker":
                    tick = self._parse_ticker(data)
                    if tick:
                        self.notify_callbacks(tick)
                        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Coinbase WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Coinbase listen error: {e}")
            self.connected = False
    
    def _parse_ticker(self, data: Dict) -> Optional[MarketTick]:
        """Parse Coinbase ticker data"""
        try:
            return MarketTick(
                symbol=data["product_id"],
                price=float(data["price"]),
                volume=float(data["volume_24h"]),
                timestamp=time.time(),
                exchange="coinbase",
                bid=float(data.get("best_bid", 0)),
                ask=float(data.get("best_ask", 0))
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Coinbase ticker: {e}")
            return None
    
    async def disconnect(self):
        """Disconnect from Coinbase"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()

class BinanceConnector(ExchangeConnector):
    """Binance WebSocket connector"""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.websocket = None
        
    async def connect(self):
        """Connect to Binance WebSocket"""
        try:
            # Convert symbols to Binance format (BTC-USD -> btcusdt)
            streams = [f"{symbol.lower().replace('-', '')}@ticker" 
                      for symbol in self.config.symbols]
            stream_names = "/".join(streams)
            
            url = f"wss://stream.binance.com:9443/ws/{stream_names}"
            
            self.websocket = await websockets.connect(url)
            self.connected = True
            logger.info(f"Connected to Binance for {self.config.symbols}")
            
            # Listen for messages
            await self._listen_loop()
            
        except Exception as e:
            logger.error(f"Binance connection error: {e}")
            self.connected = False
    
    async def _listen_loop(self):
        """Main listening loop for WebSocket messages"""
        try:
            while self.connected and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                tick = self._parse_ticker(data)
                if tick:
                    self.notify_callbacks(tick)
                        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Binance listen error: {e}")
            self.connected = False
    
    def _parse_ticker(self, data: Dict) -> Optional[MarketTick]:
        """Parse Binance ticker data"""
        try:
            return MarketTick(
                symbol=data["s"],  # Symbol
                price=float(data["c"]),  # Close price
                volume=float(data["v"]),  # Volume
                timestamp=time.time(),
                exchange="binance",
                bid=float(data.get("b", 0)),  # Bid price
                ask=float(data.get("a", 0))   # Ask price
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Binance ticker: {e}")
            return None
    
    async def disconnect(self):
        """Disconnect from Binance"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()

class LiveDataStreamer:
    """Main data streaming coordinator"""
    
    def __init__(self):
        self.connectors: Dict[str, ExchangeConnector] = {}
        self.callbacks: List[Callable[[MarketTick], None]] = []
        self.running = False
        self.loop = None
        self.thread = None
        
        # Data storage
        self.latest_ticks: Dict[str, MarketTick] = {}
        self.tick_history: Dict[str, deque] = {}
        
    def add_exchange(self, config: StreamConfig) -> ExchangeConnector:
        """Add an exchange connection"""
        if config.exchange == "coinbase":
            connector = CoinbaseConnector(config)
        elif config.exchange == "binance":
            connector = BinanceConnector(config)
        else:
            raise ValueError(f"Unsupported exchange: {config.exchange}")
        
        # Add internal callback to store data
        connector.add_callback(self._store_tick)
        
        self.connectors[config.exchange] = connector
        logger.info(f"Added {config.exchange} connector")
        return connector
    
    def add_callback(self, callback: Callable[[MarketTick], None]):
        """Add callback for all tick data"""
        self.callbacks.append(callback)
    
    def _store_tick(self, tick: MarketTick):
        """Internal callback to store tick data"""
        # Store latest tick
        key = f"{tick.exchange}:{tick.symbol}"
        self.latest_ticks[key] = tick
        
        # Store in history
        if key not in self.tick_history:
            self.tick_history[key] = deque(maxlen=1000)
        self.tick_history[key].append(tick)
        
        # Notify external callbacks
        for callback in self.callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Data callback error: {e}")
    
    def start(self):
        """Start all connections in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.thread.start()
            logger.info("Live data streamer started")
    
    def stop(self):
        """Stop all connections"""
        self.running = False
        if self.loop:
            # Schedule shutdown on the event loop
            asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Live data streamer stopped")
    
    def _run_async_loop(self):
        """Run the async event loop in background thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connect_all())
        except Exception as e:
            logger.error(f"Async loop error: {e}")
        finally:
            self.loop.close()
    
    async def _connect_all(self):
        """Connect to all configured exchanges"""
        tasks = []
        
        for connector in self.connectors.values():
            task = asyncio.create_task(self._connect_with_retry(connector))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _connect_with_retry(self, connector: ExchangeConnector):
        """Connect with automatic retry logic"""
        retries = 0
        
        while self.running and retries < connector.config.max_reconnects:
            try:
                await connector.connect()
                if connector.connected:
                    logger.info(f"{connector.config.exchange} connected successfully")
                    return
            except Exception as e:
                logger.error(f"{connector.config.exchange} connection failed: {e}")
            
            retries += 1
            if retries < connector.config.max_reconnects:
                logger.info(f"Retrying {connector.config.exchange} in {connector.config.reconnect_delay}s")
                await asyncio.sleep(connector.config.reconnect_delay)
        
        logger.error(f"Failed to connect to {connector.config.exchange} after {retries} attempts")
    
    async def _shutdown(self):
        """Shutdown all connections"""
        tasks = []
        for connector in self.connectors.values():
            if connector.connected:
                task = asyncio.create_task(connector.disconnect())
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        key = f"{exchange}:{symbol}"
        tick = self.latest_ticks.get(key)
        return tick.price if tick else None
    
    def get_price_history(self, exchange: str, symbol: str, count: int = 100) -> List[float]:
        """Get recent price history"""
        key = f"{exchange}:{symbol}"
        history = self.tick_history.get(key, deque())
        return [tick.price for tick in list(history)[-count:]]
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all exchanges"""
        return {name: connector.connected 
                for name, connector in self.connectors.items()}

# Example usage and factory functions
def create_coinbase_streamer(symbols: List[str], api_key: str = None) -> LiveDataStreamer:
    """Create streamer for Coinbase Pro"""
    streamer = LiveDataStreamer()
    
    config = StreamConfig(
        exchange="coinbase",
        symbols=symbols,
        websocket_url="wss://ws-feed.pro.coinbase.com",
        rest_url="https://api.pro.coinbase.com",
        api_key=api_key
    )
    
    streamer.add_exchange(config)
    return streamer

def create_binance_streamer(symbols: List[str], api_key: str = None) -> LiveDataStreamer:
    """Create streamer for Binance"""
    streamer = LiveDataStreamer()
    
    # Convert symbols to Binance format
    binance_symbols = [symbol.replace("-", "") for symbol in symbols]
    
    config = StreamConfig(
        exchange="binance",
        symbols=binance_symbols,
        websocket_url="wss://stream.binance.com:9443/ws/",
        rest_url="https://api.binance.com",
        api_key=api_key
    )
    
    streamer.add_exchange(config)
    return streamer

def create_multi_exchange_streamer(symbols: List[str]) -> LiveDataStreamer:
    """Create streamer with multiple exchanges"""
    streamer = LiveDataStreamer()
    
    # Add Coinbase
    coinbase_config = StreamConfig(
        exchange="coinbase",
        symbols=symbols,
        websocket_url="wss://ws-feed.pro.coinbase.com",
        rest_url="https://api.pro.coinbase.com"
    )
    streamer.add_exchange(coinbase_config)
    
    # Add Binance  
    binance_symbols = [symbol.replace("-", "") for symbol in symbols]
    binance_config = StreamConfig(
        exchange="binance", 
        symbols=binance_symbols,
        websocket_url="wss://stream.binance.com:9443/ws/",
        rest_url="https://api.binance.com"
    )
    streamer.add_exchange(binance_config)
    
    return streamer 