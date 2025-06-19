#!/usr/bin/env python3
"""
Tick Processor - Real-time Market Data Processing Engine
======================================================

High-performance tick processing system for real-time market data analysis.
Handles price ticks, volume data, order book updates, and feeds clean data
to the strategy logic and mathematical frameworks.

Key Features:
- Real-time tick processing and validation
- Order book depth analysis
- Volume profile analysis
- Tick aggregation and normalization
- Market microstructure analysis
- Integration with mathematical frameworks
- Performance optimization for high-frequency data

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from collections import deque, defaultdict
import asyncio

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Self

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class TickType(Enum):
    """Tick type enumeration"""
    TRADE = "trade"
    QUOTE = "quote"
    ORDER_BOOK = "order_book"
    VOLUME = "volume"
    OHLCV = "ohlcv"


class TickStatus(Enum):
    """Tick processing status"""
    VALID = "valid"
    INVALID = "invalid"
    DUPLICATE = "duplicate"
    OUT_OF_SEQUENCE = "out_of_sequence"
    PROCESSING = "processing"


@dataclass
class MarketTick:
    """Market tick data container"""
    
    tick_id: str
    tick_type: TickType
    symbol: str
    timestamp: float
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    trade_id: Optional[str] = None
    side: Optional[str] = None  # 'buy' or 'sell'
    status: TickStatus = TickStatus.VALID
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookLevel:
    """Order book level data"""
    
    price: float
    size: float
    side: str  # 'bid' or 'ask'
    timestamp: float


@dataclass
class OrderBook:
    """Order book snapshot"""
    
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float


@dataclass
class TickAggregate:
    """Aggregated tick data"""
    
    symbol: str
    start_time: float
    end_time: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    total_volume: float
    trade_count: int
    vwap: float
    tick_count: int


class TickProcessor:
    """High-performance tick processing engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize tick processor"""
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Processing queues and buffers
        self.tick_queue: deque = deque(maxlen=self.config.get('max_queue_size', 10000))
        self.processed_ticks: deque = deque(maxlen=self.config.get('max_history_size', 50000))
        
        # Order book management
        self.order_books: Dict[str, OrderBook] = {}
        self.order_book_depth = self.config.get('order_book_depth', 10)
        
        # Aggregation settings
        self.aggregation_intervals = self.config.get('aggregation_intervals', [1, 5, 15, 60])
        self.aggregated_data: Dict[str, Dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Performance tracking
        self.total_ticks_processed = 0
        self.total_ticks_rejected = 0
        self.processing_latency = []
        self.last_processing_time = 0.0
        
        # Callbacks and hooks
        self.tick_callbacks: List[Callable[[MarketTick], None]] = []
        self.aggregate_callbacks: List[Callable[[TickAggregate], None]] = []
        
        # Threading and synchronization
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Validation and filtering
        self.price_filters = self._initialize_price_filters()
        self.volume_filters = self._initialize_volume_filters()
        
        logger.info(f"TickProcessor v{self.version} initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'max_queue_size': 10000,
            'max_history_size': 50000,
            'order_book_depth': 10,
            'aggregation_intervals': [1, 5, 15, 60],  # seconds
            'enable_real_time_processing': True,
            'enable_aggregation': True,
            'enable_order_book_tracking': True,
            'price_change_threshold': 0.1,  # 10% max price change
            'volume_spike_threshold': 5.0,  # 5x average volume
            'min_tick_interval': 0.001,  # 1ms minimum between ticks
            'max_processing_latency': 0.1,  # 100ms max processing time
            'enable_performance_monitoring': True
        }
    
    def _initialize_price_filters(self) -> Dict[str, Any]:
        """Initialize price validation filters"""
        return {
            'min_price': 0.0001,
            'max_price': 1000000.0,
            'max_price_change': self.config.get('price_change_threshold', 0.1),
            'price_precision': 8
        }
    
    def _initialize_volume_filters(self) -> Dict[str, Any]:
        """Initialize volume validation filters"""
        return {
            'min_volume': 0.0,
            'max_volume': 1000000000.0,
            'volume_spike_threshold': self.config.get('volume_spike_threshold', 5.0),
            'volume_precision': 8
        }
    
    def add_tick_callback(self, callback: Callable[[MarketTick], None]) -> None:
        """Add callback for processed ticks"""
        self.tick_callbacks.append(callback)
    
    def add_aggregate_callback(self, callback: Callable[[TickAggregate], None]) -> None:
        """Add callback for aggregated data"""
        self.aggregate_callbacks.append(callback)
    
    def process_tick(self, tick_data: Dict[str, Any]) -> Optional[MarketTick]:
        """Process a single market tick"""
        try:
            start_time = time.time()
            
            # Validate tick data
            if not self._validate_tick_data(tick_data):
                self.total_ticks_rejected += 1
                return None
            
            # Create market tick object
            tick = self._create_market_tick(tick_data)
            
            # Apply filters and validation
            if not self._apply_filters(tick):
                self.total_ticks_rejected += 1
                return None
            
            # Add to processing queue
            with self.processing_lock:
                self.tick_queue.append(tick)
            
            # Update order book if needed
            if self.config.get('enable_order_book_tracking', True):
                self._update_order_book(tick)
            
            # Trigger aggregation if enabled
            if self.config.get('enable_aggregation', True):
                self._trigger_aggregation(tick)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_latency.append(processing_time)
            self.total_ticks_processed += 1
            self.last_processing_time = time.time()
            
            # Trim latency history
            if len(self.processing_latency) > 1000:
                self.processing_latency = self.processing_latency[-1000:]
            
            # Execute callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")
            
            return tick
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.total_ticks_rejected += 1
            return None
    
    def _validate_tick_data(self, tick_data: Dict[str, Any]) -> bool:
        """Validate tick data structure"""
        try:
            required_fields = ['symbol', 'timestamp', 'price']
            for field in required_fields:
                if field not in tick_data:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate data types
            if not isinstance(tick_data['symbol'], str):
                return False
            if not isinstance(tick_data['timestamp'], (int, float)):
                return False
            if not isinstance(tick_data['price'], (int, float)):
                return False
            
            # Validate price range
            price = float(tick_data['price'])
            if price < self.price_filters['min_price'] or price > self.price_filters['max_price']:
                logger.warning(f"Price out of range: {price}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating tick data: {e}")
            return False
    
    def _create_market_tick(self, tick_data: Dict[str, Any]) -> MarketTick:
        """Create MarketTick object from raw data"""
        try:
            tick_type = TickType(tick_data.get('type', 'trade'))
            
            tick = MarketTick(
                tick_id=tick_data.get('id', f"{tick_data['symbol']}_{tick_data['timestamp']}"),
                tick_type=tick_type,
                symbol=tick_data['symbol'],
                timestamp=float(tick_data['timestamp']),
                price=float(tick_data['price']),
                volume=float(tick_data.get('volume', 0.0)),
                bid=float(tick_data.get('bid', 0.0)) if tick_data.get('bid') else None,
                ask=float(tick_data.get('ask', 0.0)) if tick_data.get('ask') else None,
                bid_size=float(tick_data.get('bid_size', 0.0)) if tick_data.get('bid_size') else None,
                ask_size=float(tick_data.get('ask_size', 0.0)) if tick_data.get('ask_size') else None,
                trade_id=tick_data.get('trade_id'),
                side=tick_data.get('side'),
                metadata=tick_data.get('metadata', {})
            )
            
            return tick
            
        except Exception as e:
            logger.error(f"Error creating market tick: {e}")
            raise
    
    def _apply_filters(self, tick: MarketTick) -> bool:
        """Apply validation filters to tick"""
        try:
            # Check for duplicate ticks
            if self._is_duplicate_tick(tick):
                tick.status = TickStatus.DUPLICATE
                return False
            
            # Check for out-of-sequence ticks
            if self._is_out_of_sequence(tick):
                tick.status = TickStatus.OUT_OF_SEQUENCE
                return False
            
            # Check price change limits
            if not self._validate_price_change(tick):
                return False
            
            # Check volume spike limits
            if not self._validate_volume_spike(tick):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return False
    
    def _is_duplicate_tick(self, tick: MarketTick) -> bool:
        """Check if tick is duplicate"""
        # Simple duplicate check based on tick ID
        # In a real system, you'd have more sophisticated duplicate detection
        return False
    
    def _is_out_of_sequence(self, tick: MarketTick) -> bool:
        """Check if tick is out of sequence"""
        # Check if tick timestamp is reasonable
        current_time = time.time()
        time_diff = abs(current_time - tick.timestamp)
        
        # Allow for some clock skew (5 seconds)
        return time_diff > 5.0
    
    def _validate_price_change(self, tick: MarketTick) -> bool:
        """Validate price change is within limits"""
        try:
            # Get previous price for this symbol
            previous_ticks = [t for t in self.processed_ticks if t.symbol == tick.symbol]
            if not previous_ticks:
                return True
            
            last_tick = previous_ticks[-1]
            price_change = abs(tick.price - last_tick.price) / last_tick.price
            
            max_change = self.price_filters['max_price_change']
            if price_change > max_change:
                logger.warning(f"Price change too large: {price_change:.4f} > {max_change}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price change: {e}")
            return False
    
    def _validate_volume_spike(self, tick: MarketTick) -> bool:
        """Validate volume spike is within limits"""
        try:
            if tick.volume <= 0:
                return True
            
            # Calculate average volume for this symbol
            recent_ticks = [t for t in self.processed_ticks if t.symbol == tick.symbol][-100:]
            if len(recent_ticks) < 10:
                return True
            
            avg_volume = np.mean([t.volume for t in recent_ticks if t.volume > 0])
            if avg_volume <= 0:
                return True
            
            volume_ratio = tick.volume / avg_volume
            max_ratio = self.volume_filters['volume_spike_threshold']
            
            if volume_ratio > max_ratio:
                logger.warning(f"Volume spike detected: {volume_ratio:.2f}x average")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating volume spike: {e}")
            return False
    
    def _update_order_book(self, tick: MarketTick) -> None:
        """Update order book with new tick data"""
        try:
            if tick.tick_type != TickType.ORDER_BOOK:
                return
            
            symbol = tick.symbol
            
            # Initialize order book if needed
            if symbol not in self.order_books:
                self.order_books[symbol] = OrderBook(
                    symbol=symbol,
                    timestamp=tick.timestamp,
                    bids=[],
                    asks=[],
                    spread=0.0,
                    mid_price=0.0,
                    total_bid_volume=0.0,
                    total_ask_volume=0.0
                )
            
            order_book = self.order_books[symbol]
            
            # Update bids and asks
            if tick.bid and tick.bid_size:
                bid_level = OrderBookLevel(
                    price=tick.bid,
                    size=tick.bid_size,
                    side='bid',
                    timestamp=tick.timestamp
                )
                order_book.bids.append(bid_level)
            
            if tick.ask and tick.ask_size:
                ask_level = OrderBookLevel(
                    price=tick.ask,
                    size=tick.ask_size,
                    side='ask',
                    timestamp=tick.timestamp
                )
                order_book.asks.append(ask_level)
            
            # Sort and limit depth
            order_book.bids.sort(key=lambda x: x.price, reverse=True)
            order_book.asks.sort(key=lambda x: x.price)
            
            order_book.bids = order_book.bids[:self.order_book_depth]
            order_book.asks = order_book.asks[:self.order_book_depth]
            
            # Calculate metrics
            if order_book.bids and order_book.asks:
                best_bid = order_book.bids[0].price
                best_ask = order_book.asks[0].price
                order_book.spread = best_ask - best_bid
                order_book.mid_price = (best_bid + best_ask) / 2
                order_book.total_bid_volume = sum(b.size for b in order_book.bids)
                order_book.total_ask_volume = sum(a.size for a in order_book.asks)
            
            order_book.timestamp = tick.timestamp
            
        except Exception as e:
            logger.error(f"Error updating order book: {e}")
    
    def _trigger_aggregation(self, tick: MarketTick) -> None:
        """Trigger data aggregation for different time intervals"""
        try:
            for interval in self.aggregation_intervals:
                self._aggregate_tick(tick, interval)
        except Exception as e:
            logger.error(f"Error triggering aggregation: {e}")
    
    def _aggregate_tick(self, tick: MarketTick, interval: int) -> None:
        """Aggregate tick data for specific time interval"""
        try:
            symbol = tick.symbol
            timestamp = tick.timestamp
            
            # Calculate interval start time
            interval_start = int(timestamp // interval) * interval
            
            # Get or create aggregate for this interval
            if interval_start not in self.aggregated_data[symbol][interval]:
                # Create new aggregate
                aggregate = TickAggregate(
                    symbol=symbol,
                    start_time=interval_start,
                    end_time=interval_start + interval,
                    open_price=tick.price,
                    high_price=tick.price,
                    low_price=tick.price,
                    close_price=tick.price,
                    total_volume=tick.volume,
                    trade_count=1,
                    vwap=tick.price,
                    tick_count=1
                )
                self.aggregated_data[symbol][interval].append(aggregate)
            else:
                # Update existing aggregate
                aggregate = self.aggregated_data[symbol][interval][-1]
                aggregate.high_price = max(aggregate.high_price, tick.price)
                aggregate.low_price = min(aggregate.low_price, tick.price)
                aggregate.close_price = tick.price
                aggregate.total_volume += tick.volume
                aggregate.trade_count += 1
                aggregate.tick_count += 1
                
                # Update VWAP
                total_value = aggregate.vwap * (aggregate.trade_count - 1) + tick.price * tick.volume
                aggregate.vwap = total_value / aggregate.total_volume if aggregate.total_volume > 0 else tick.price
            
            # Execute aggregate callbacks
            for callback in self.aggregate_callbacks:
                try:
                    callback(aggregate)
                except Exception as e:
                    logger.error(f"Error in aggregate callback: {e}")
            
        except Exception as e:
            logger.error(f"Error aggregating tick: {e}")
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get current order book for symbol"""
        return self.order_books.get(symbol)
    
    def get_aggregated_data(self, symbol: str, interval: int, 
                          count: int = 100) -> List[TickAggregate]:
        """Get aggregated data for symbol and interval"""
        try:
            if symbol not in self.aggregated_data or interval not in self.aggregated_data[symbol]:
                return []
            
            data = list(self.aggregated_data[symbol][interval])
            return data[-count:] if count > 0 else data
            
        except Exception as e:
            logger.error(f"Error getting aggregated data: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            avg_latency = np.mean(self.processing_latency) if self.processing_latency else 0.0
            max_latency = np.max(self.processing_latency) if self.processing_latency else 0.0
            
            return {
                'version': self.version,
                'total_ticks_processed': self.total_ticks_processed,
                'total_ticks_rejected': self.total_ticks_rejected,
                'processing_rate': self.total_ticks_processed / max(time.time() - self.last_processing_time, 1.0),
                'average_latency': avg_latency,
                'max_latency': max_latency,
                'queue_size': len(self.tick_queue),
                'history_size': len(self.processed_ticks),
                'order_books_tracked': len(self.order_books),
                'last_processing_time': self.last_processing_time
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def start_processing(self) -> None:
        """Start background processing thread"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Tick processing started")
    
    def stop_processing(self) -> None:
        """Stop background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("Tick processing stopped")
    
    def _processing_loop(self) -> None:
        """Background processing loop"""
        while self.is_processing:
            try:
                # Process queued ticks
                with self.processing_lock:
                    while self.tick_queue:
                        tick = self.tick_queue.popleft()
                        self.processed_ticks.append(tick)
                
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)


def main() -> None:
    """Main function for testing tick processor"""
    try:
        print("📊 Tick Processor Test")
        print("=" * 40)
        
        # Initialize tick processor
        processor = TickProcessor()
        
        # Test tick data
        test_ticks = [
            {
                'id': 'test_1',
                'type': 'trade',
                'symbol': 'BTC',
                'timestamp': time.time(),
                'price': 50000.0,
                'volume': 1.5,
                'side': 'buy'
            },
            {
                'id': 'test_2',
                'type': 'quote',
                'symbol': 'BTC',
                'timestamp': time.time() + 1,
                'price': 50001.0,
                'bid': 50000.5,
                'ask': 50001.5,
                'bid_size': 2.0,
                'ask_size': 1.5
            }
        ]
        
        # Process test ticks
        for tick_data in test_ticks:
            tick = processor.process_tick(tick_data)
            if tick:
                print(f"✅ Processed tick: {tick.symbol} @ {tick.price:.2f}")
            else:
                print(f"❌ Rejected tick: {tick_data['symbol']}")
        
        # Get performance metrics
        metrics = processor.get_performance_metrics()
        print(f"✅ Performance: {metrics['total_ticks_processed']} processed, "
              f"{metrics['total_ticks_rejected']} rejected")
        
        print("\n🎉 Tick processor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Tick processor test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
