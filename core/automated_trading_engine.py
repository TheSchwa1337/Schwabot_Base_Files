#!/usr/bin/env python3
"""
Automated Trading Engine - Core CCXT Integration
Handles automated trading with batch orders, buy/sell walls, and real-time price tracking
"""

import ccxt
import asyncio
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from queue import Queue
import json

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal with automated execution parameters."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None  # None for market orders
    order_type: str = 'market'  # 'market', 'limit', 'stop'
    batch_size: int = 1  # Number of orders in batch
    spread_seconds: int = 0  # Spread orders across time
    strategy_id: str = 'automated'
    confidence: float = 0.8
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class BatchOrder:
    """Batch order configuration for automated trading."""
    symbol: str
    side: str
    total_quantity: float
    batch_count: int  # 1-50 orders per batch
    price_range: Tuple[float, float]  # Min/max price for limit orders
    spread_seconds: int  # Time spread between orders
    strategy: str
    priority: int = 1

class AutomatedTradingEngine:
    """Core automated trading engine with CCXT integration."""
    
    def __init__(self, exchange_config: Dict, api_key: str = None, secret: str = None):
        """
        Initialize automated trading engine.
        
        Args:
            exchange_config: Exchange configuration
            api_key: API key for trading
            secret: Secret key for trading
        """
        self.exchange_config = exchange_config
        self.api_key = api_key
        self.secret = secret
        
        # Initialize CCXT exchange
        self.exchange = self._initialize_exchange()
        
        # Trading state
        self.active_orders = {}
        self.order_history = []
        self.portfolio = {}
        self.price_cache = {}
        
        # Batch order queue
        self.batch_queue = Queue()
        self.batch_processor = None
        
        # Real-time price tracking
        self.price_trackers = {}
        self.tracking_symbols = set()
        
        # Mathematical tensor state
        self.tensor_state = {
            'momentum': {},
            'volatility': {},
            'correlation_matrix': {},
            'basket_weights': {}
        }
        
        # Start background processors
        self._start_background_processors()
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange with proper configuration."""
        exchange_name = self.exchange_config.get('name', 'coinbase')
        
        # Exchange class mapping
        exchange_map = {
            'coinbase': ccxt.coinbase,
            'binance': ccxt.binance,
            'kraken': ccxt.kraken,
            'kucoin': ccxt.kucoin
        }
        
        exchange_class = exchange_map.get(exchange_name.lower(), ccxt.coinbase)
        
        # Initialize exchange
        exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.secret,
            'sandbox': self.exchange_config.get('sandbox', False),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # or 'future' for futures trading
            }
        })
        
        logger.info(f"Initialized {exchange_name} exchange for automated trading")
        return exchange
    
    def _start_background_processors(self):
        """Start background processors for automated trading."""
        # Start batch order processor
        self.batch_processor = threading.Thread(target=self._process_batch_orders, daemon=True)
        self.batch_processor.start()
        
        # Start price tracker
        self.price_tracker = threading.Thread(target=self._track_prices, daemon=True)
        self.price_tracker.start()
        
        logger.info("Started background processors for automated trading")
    
    def add_symbol_to_tracking(self, symbol: str):
        """Add symbol to real-time price tracking."""
        self.tracking_symbols.add(symbol)
        logger.info(f"Added {symbol} to price tracking")
    
    def remove_symbol_from_tracking(self, symbol: str):
        """Remove symbol from real-time price tracking."""
        self.tracking_symbols.discard(symbol)
        logger.info(f"Removed {symbol} from price tracking")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        return self.price_cache.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices."""
        return self.price_cache.copy()
    
    def _track_prices(self):
        """Background thread for real-time price tracking."""
        while True:
            try:
                for symbol in self.tracking_symbols:
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        self.price_cache[symbol] = ticker['last']
                        
                        # Update tensor state with new price
                        self._update_tensor_state(symbol, ticker['last'])
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch price for {symbol}: {e}")
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in price tracking: {e}")
                time.sleep(5)
    
    def _update_tensor_state(self, symbol: str, price: float):
        """Update mathematical tensor state with new price data."""
        if symbol not in self.tensor_state['momentum']:
            self.tensor_state['momentum'][symbol] = []
            self.tensor_state['volatility'][symbol] = []
        
        # Add price to momentum calculation
        momentum_data = self.tensor_state['momentum'][symbol]
        momentum_data.append(price)
        
        # Keep last 100 prices for calculations
        if len(momentum_data) > 100:
            momentum_data.pop(0)
        
        # Calculate momentum
        if len(momentum_data) > 1:
            momentum = (momentum_data[-1] - momentum_data[-2]) / momentum_data[-2]
            self.tensor_state['momentum'][symbol] = momentum
        
        # Calculate volatility (rolling standard deviation)
        if len(momentum_data) > 10:
            recent_prices = momentum_data[-10:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            self.tensor_state['volatility'][symbol] = volatility
    
    def create_buy_wall(self, symbol: str, total_quantity: float, price_range: Tuple[float, float], 
                       batch_count: int = 10, spread_seconds: int = 30) -> str:
        """
        Create automated buy wall with batch orders.
        
        Args:
            symbol: Trading symbol
            total_quantity: Total quantity to buy
            price_range: (min_price, max_price) for limit orders
            batch_count: Number of orders in batch (1-50)
            spread_seconds: Time to spread orders across
        
        Returns:
            Batch order ID
        """
        batch_id = f"buy_wall_{symbol}_{int(time.time())}"
        
        batch_order = BatchOrder(
            symbol=symbol,
            side='buy',
            total_quantity=total_quantity,
            batch_count=min(batch_count, 50),  # Cap at 50 orders
            price_range=price_range,
            spread_seconds=spread_seconds,
            strategy='buy_wall'
        )
        
        self.batch_queue.put((batch_id, batch_order))
        logger.info(f"Created buy wall {batch_id} for {symbol}")
        
        return batch_id
    
    def create_sell_wall(self, symbol: str, total_quantity: float, price_range: Tuple[float, float], 
                        batch_count: int = 10, spread_seconds: int = 30) -> str:
        """
        Create automated sell wall with batch orders.
        
        Args:
            symbol: Trading symbol
            total_quantity: Total quantity to sell
            price_range: (min_price, max_price) for limit orders
            batch_count: Number of orders in batch (1-50)
            spread_seconds: Time to spread orders across
        
        Returns:
            Batch order ID
        """
        batch_id = f"sell_wall_{symbol}_{int(time.time())}"
        
        batch_order = BatchOrder(
            symbol=symbol,
            side='sell',
            total_quantity=total_quantity,
            batch_count=min(batch_count, 50),  # Cap at 50 orders
            price_range=price_range,
            spread_seconds=spread_seconds,
            strategy='sell_wall'
        )
        
        self.batch_queue.put((batch_id, batch_order))
        logger.info(f"Created sell wall {batch_id} for {symbol}")
        
        return batch_id
    
    def create_basket_order(self, basket_symbols: List[str], weights: List[float], 
                           total_value: float, strategy: str = 'basket') -> str:
        """
        Create automated basket order across multiple symbols.
        
        Args:
            basket_symbols: List of symbols to trade
            weights: Weight for each symbol (should sum to 1.0)
            total_value: Total USD value to trade
            strategy: Strategy identifier
        
        Returns:
            Basket order ID
        """
        basket_id = f"basket_{strategy}_{int(time.time())}"
        
        # Calculate quantities based on current prices and weights
        quantities = []
        for symbol, weight in zip(basket_symbols, weights):
            current_price = self.get_current_price(symbol)
            if current_price:
                quantity = (total_value * weight) / current_price
                quantities.append(quantity)
            else:
                quantities.append(0)
        
        # Create individual orders for each symbol
        for symbol, quantity in zip(basket_symbols, quantities):
            if quantity > 0:
                signal = TradingSignal(
                    symbol=symbol,
                    side='buy',
                    quantity=quantity,
                    strategy_id=strategy,
                    batch_size=1
                )
                self._execute_signal(signal)
        
        logger.info(f"Created basket order {basket_id} for {len(basket_symbols)} symbols")
        return basket_id
    
    def _process_batch_orders(self):
        """Background thread for processing batch orders."""
        while True:
            try:
                if not self.batch_queue.empty():
                    batch_id, batch_order = self.batch_queue.get()
                    self._execute_batch_order(batch_id, batch_order)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing batch orders: {e}")
                time.sleep(1)
    
    def _execute_batch_order(self, batch_id: str, batch_order: BatchOrder):
        """Execute a batch order by creating multiple individual orders."""
        try:
            # Calculate order parameters
            quantity_per_order = batch_order.total_quantity / batch_order.batch_count
            time_between_orders = batch_order.spread_seconds / batch_order.batch_count
            
            # Create orders
            for i in range(batch_order.batch_count):
                # Calculate price for this order
                if batch_order.price_range[0] == batch_order.price_range[1]:
                    price = batch_order.price_range[0]
                else:
                    # Distribute prices across range
                    price_ratio = i / (batch_order.batch_count - 1) if batch_order.batch_count > 1 else 0.5
                    price = batch_order.price_range[0] + (batch_order.price_range[1] - batch_order.price_range[0]) * price_ratio
                
                # Create trading signal
                signal = TradingSignal(
                    symbol=batch_order.symbol,
                    side=batch_order.side,
                    quantity=quantity_per_order,
                    price=price,
                    order_type='limit',
                    batch_size=1,
                    strategy_id=batch_order.strategy
                )
                
                # Execute order
                order_id = self._execute_signal(signal)
                
                # Store order info
                self.active_orders[order_id] = {
                    'batch_id': batch_id,
                    'signal': signal,
                    'status': 'pending'
                }
                
                # Wait before next order
                if i < batch_order.batch_count - 1:
                    time.sleep(time_between_orders)
            
            logger.info(f"Executed batch order {batch_id} with {batch_order.batch_count} orders")
            
        except Exception as e:
            logger.error(f"Error executing batch order {batch_id}: {e}")
    
    def _execute_signal(self, signal: TradingSignal) -> str:
        """Execute a single trading signal."""
        try:
            # Prepare order parameters
            order_params = {
                'symbol': signal.symbol,
                'type': signal.order_type,
                'side': signal.side,
                'amount': signal.quantity,
            }
            
            if signal.price and signal.order_type == 'limit':
                order_params['price'] = signal.price
            
            # Execute order
            order = self.exchange.create_order(**order_params)
            
            # Store order info
            order_id = order['id']
            self.active_orders[order_id] = {
                'signal': signal,
                'order': order,
                'status': 'pending',
                'timestamp': datetime.now()
            }
            
            logger.info(f"Executed {signal.side} order {order_id} for {signal.quantity} {signal.symbol}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            raise
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of a specific order."""
        if order_id in self.active_orders:
            try:
                # Fetch updated order status from exchange
                order = self.exchange.fetch_order(order_id)
                self.active_orders[order_id]['order'] = order
                self.active_orders[order_id]['status'] = order['status']
                
                # Move to history if completed
                if order['status'] in ['closed', 'canceled']:
                    self.order_history.append(self.active_orders[order_id])
                    del self.active_orders[order_id]
                
                return self.active_orders[order_id]
            except Exception as e:
                logger.warning(f"Could not fetch order status for {order_id}: {e}")
                return self.active_orders.get(order_id, {})
        
        return {}
    
    def get_all_orders(self) -> Dict:
        """Get all active orders."""
        return self.active_orders.copy()
    
    def get_order_history(self) -> List[Dict]:
        """Get order history."""
        return self.order_history.copy()
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        try:
            self.exchange.cancel_order(order_id)
            if order_id in self.active_orders:
                self.active_orders[order_id]['status'] = 'canceled'
                self.order_history.append(self.active_orders[order_id])
                del self.active_orders[order_id]
            
            logger.info(f"Canceled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def get_portfolio(self) -> Dict:
        """Get current portfolio balances."""
        try:
            balance = self.exchange.fetch_balance()
            self.portfolio = balance
            return balance
        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return self.portfolio
    
    def get_tensor_state(self) -> Dict:
        """Get current mathematical tensor state."""
        return self.tensor_state.copy()
    
    def calculate_basket_correlation(self, symbols: List[str]) -> np.ndarray:
        """Calculate correlation matrix for basket of symbols."""
        try:
            # Get price data for all symbols
            price_data = []
            for symbol in symbols:
                if symbol in self.tensor_state['momentum']:
                    prices = self.tensor_state['momentum'][symbol]
                    if len(prices) > 10:
                        price_data.append(prices[-10:])
            
            if len(price_data) > 1:
                # Calculate correlation matrix
                price_matrix = np.array(price_data)
                correlation_matrix = np.corrcoef(price_matrix)
                return correlation_matrix
            else:
                return np.array([])
                
        except Exception as e:
            logger.error(f"Error calculating basket correlation: {e}")
            return np.array([])
    
    def optimize_basket_weights(self, symbols: List[str], target_volatility: float = 0.1) -> List[float]:
        """
        Optimize basket weights based on mathematical tensor analysis.
        
        Args:
            symbols: List of symbols in basket
            target_volatility: Target portfolio volatility
        
        Returns:
            Optimized weights for each symbol
        """
        try:
            # Get current volatilities
            volatilities = []
            for symbol in symbols:
                vol = self.tensor_state['volatility'].get(symbol, 0.1)
                volatilities.append(vol)
            
            # Simple equal-weight optimization
            # In a real system, this would use more sophisticated optimization
            weights = [1.0 / len(symbols)] * len(symbols)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error optimizing basket weights: {e}")
            return [1.0 / len(symbols)] * len(symbols)
    
    def shutdown(self):
        """Shutdown the automated trading engine."""
        logger.info("Shutting down automated trading engine...")
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            self.cancel_order(order_id)
        
        # Stop background processors
        # Note: In a real implementation, you'd want proper thread shutdown
        logger.info("Automated trading engine shutdown complete") 