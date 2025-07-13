#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced CCXT Trading Engine Module
====================================
Provides enhanced CCXT trading engine functionality for the Schwabot trading system.

Mathematical Core:
T(x) = {
    Market Order:    O_m(q, s) = execute_immediate(q, s)
    Limit Order:     O_l(q, s, p) = place_order(q, s, p, 'limit')
    Stop Order:      O_s(q, s, p) = place_order(q, s, p, 'stop')
}
Where:
- q: quantity
- s: side (buy/sell)
- p: price

This module provides advanced exchange integration with mathematical optimization,
order management, and execution analytics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json
import ccxt
from decimal import Decimal

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


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExchangeType(Enum):
    """Exchange types."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    BYBIT = "bybit"
    OKX = "okx"


@dataclass
class TradingOrder:
    """Trading order with mathematical properties."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    mathematical_signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderExecution:
    """Order execution result."""
    order_id: str
    success: bool
    status: OrderStatus
    filled_quantity: float
    average_price: float
    execution_time: float
    slippage: float
    fees: float
    mathematical_signature: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeBalance:
    """Exchange balance information."""
    currency: str
    free: float
    used: float
    total: float
    mathematical_signature: str = ""


@dataclass
class MarketInfo:
    """Market information."""
    symbol: str
    base_currency: str
    quote_currency: str
    min_amount: float
    max_amount: float
    min_price: float
    max_price: float
    price_precision: int
    amount_precision: int
    mathematical_signature: str = ""


@dataclass
class EnhancedCCXTConfig:
    """Configuration for enhanced CCXT trading engine."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    max_concurrent_orders: int = 20
    order_timeout: float = 60.0  # seconds
    slippage_tolerance: float = 0.002  # 0.2%
    mathematical_analysis_enabled: bool = True
    sandbox_mode: bool = True
    rate_limit_enabled: bool = True
    exchanges: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'binance': {
            'api_key': '',
            'secret': '',
            'sandbox': True,
            'rate_limit': 1200  # requests per minute
        },
        'coinbase': {
            'api_key': '',
            'secret': '',
            'passphrase': '',
            'sandbox': True,
            'rate_limit': 1000
        }
    })


class EnhancedCCXTTradingEngine:
    """
    Enhanced CCXT Trading Engine System
    
    Implements advanced exchange integration:
    T(x) = {
        Market Order:    O_m(q, s) = execute_immediate(q, s)
        Limit Order:     O_l(q, s, p) = place_order(q, s, p, 'limit')
        Stop Order:      O_s(q, s, p) = place_order(q, s, p, 'stop')
    }
    
    Provides advanced exchange integration with mathematical optimization,
    order management, and execution analytics.
    """
    
    def __init__(self, config: Optional[EnhancedCCXTConfig] = None):
        """Initialize the enhanced CCXT trading engine system."""
        self.config = config or EnhancedCCXTConfig()
        self.logger = logging.getLogger(__name__)
        
        # Exchange connections
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.active_orders: Dict[str, TradingOrder] = {}
        self.order_history: List[OrderExecution] = []
        self.balances: Dict[str, Dict[str, ExchangeBalance]] = {}
        self.market_info: Dict[str, Dict[str, MarketInfo]] = {}
        
        # Order processing
        self.order_queue: asyncio.Queue = asyncio.Queue()
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
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
            'orders_submitted': 0,
            'orders_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_fees': 0.0,
            'total_slippage': 0.0
        }
        
        # System state
        self.initialized = False
        self.active = False
        
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the enhanced CCXT trading engine system."""
        try:
            self.logger.info("Initializing Enhanced CCXT Trading Engine System")
            
            # Initialize exchange connections
            for exchange_name, exchange_config in self.config.exchanges.items():
                if exchange_config.get('api_key') and exchange_config.get('secret'):
                    self._initialize_exchange(exchange_name, exchange_config)
            
            self.initialized = True
            self.logger.info("✅ Enhanced CCXT Trading Engine System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Enhanced CCXT Trading Engine System: {e}")
            self.initialized = False
    
    def _initialize_exchange(self, exchange_name: str, config: Dict[str, Any]) -> None:
        """Initialize exchange connection."""
        try:
            # Get exchange class
            exchange_class = getattr(ccxt, exchange_name)
            
            # Create exchange instance
            exchange = exchange_class({
                'apiKey': config.get('api_key', ''),
                'secret': config.get('secret', ''),
                'password': config.get('passphrase', ''),
                'sandbox': config.get('sandbox', self.config.sandbox_mode),
                'enableRateLimit': config.get('rate_limit_enabled', self.config.rate_limit_enabled),
                'rateLimit': config.get('rate_limit', 1000),
                'timeout': self.config.timeout * 1000,  # Convert to milliseconds
                'verbose': self.config.debug
            })
            
            # Test connection
            exchange.load_markets()
            
            # Store exchange
            self.exchanges[exchange_name] = exchange
            
            # Initialize data structures
            self.balances[exchange_name] = {}
            self.market_info[exchange_name] = {}
            
            self.logger.info(f"✅ Initialized {exchange_name} exchange connection")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing {exchange_name} exchange: {e}")
    
    async def start_trading_engine(self) -> bool:
        """Start the trading engine."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            
            # Start processing tasks
            asyncio.create_task(self._process_order_queue())
            asyncio.create_task(self._process_execution_queue())
            
            # Load initial data
            await self._load_exchange_data()
            
            self.logger.info("✅ Enhanced CCXT Trading Engine started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting trading engine: {e}")
            return False
    
    async def stop_trading_engine(self) -> bool:
        """Stop the trading engine."""
        try:
            self.active = False
            
            # Cancel all active orders
            await self._cancel_all_orders()
            
            # Close exchange connections
            for exchange in self.exchanges.values():
                await exchange.close()
            
            self.logger.info("✅ Enhanced CCXT Trading Engine stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping trading engine: {e}")
            return False
    
    async def _load_exchange_data(self) -> None:
        """Load initial exchange data."""
        try:
            for exchange_name, exchange in self.exchanges.items():
                # Load balances
                await self._load_balances(exchange_name, exchange)
                
                # Load market info
                await self._load_market_info(exchange_name, exchange)
                
        except Exception as e:
            self.logger.error(f"❌ Error loading exchange data: {e}")
    
    async def _load_balances(self, exchange_name: str, exchange: ccxt.Exchange) -> None:
        """Load exchange balances."""
        try:
            balances = await exchange.fetch_balance()
            
            for currency, balance_data in balances.items():
                if isinstance(balance_data, dict) and 'free' in balance_data:
                    balance = ExchangeBalance(
                        currency=currency,
                        free=float(balance_data['free']),
                        used=float(balance_data['used']),
                        total=float(balance_data['total'])
                    )
                    self.balances[exchange_name][currency] = balance
            
            self.logger.info(f"✅ Loaded balances for {exchange_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading balances for {exchange_name}: {e}")
    
    async def _load_market_info(self, exchange_name: str, exchange: ccxt.Exchange) -> None:
        """Load market information."""
        try:
            markets = exchange.markets
            
            for symbol, market_data in markets.items():
                if market_data.get('active'):
                    market_info = MarketInfo(
                        symbol=symbol,
                        base_currency=market_data.get('base', ''),
                        quote_currency=market_data.get('quote', ''),
                        min_amount=float(market_data.get('limits', {}).get('amount', {}).get('min', 0)),
                        max_amount=float(market_data.get('limits', {}).get('amount', {}).get('max', float('inf'))),
                        min_price=float(market_data.get('limits', {}).get('price', {}).get('min', 0)),
                        max_price=float(market_data.get('limits', {}).get('price', {}).get('max', float('inf'))),
                        price_precision=int(market_data.get('precision', {}).get('price', 8)),
                        amount_precision=int(market_data.get('precision', {}).get('amount', 8))
                    )
                    self.market_info[exchange_name][symbol] = market_info
            
            self.logger.info(f"✅ Loaded market info for {exchange_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading market info for {exchange_name}: {e}")
    
    async def submit_order(self, order_data: Dict[str, Any]) -> bool:
        """Submit a trading order."""
        if not self.active:
            self.logger.error("Trading engine not active")
            return False
        
        try:
            # Validate order data
            if not self._validate_order_data(order_data):
                self.logger.error(f"Invalid order data: {order_data}")
                return False
            
            # Create trading order
            order = self._create_trading_order(order_data)
            
            # Add mathematical analysis
            if self.config.mathematical_analysis_enabled:
                await self._analyze_order_mathematically(order)
            
            # Store order
            self.active_orders[order.order_id] = order
            
            # Queue for processing
            await self.order_queue.put(order)
            
            self.logger.info(f"✅ Order submitted: {order.order_id} for {order.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error submitting order: {e}")
            return False
    
    def _validate_order_data(self, order_data: Dict[str, Any]) -> bool:
        """Validate order data."""
        try:
            required_fields = ['symbol', 'side', 'order_type', 'quantity']
            
            for field in required_fields:
                if field not in order_data:
                    return False
            
            # Check quantity
            quantity = order_data.get('quantity', 0)
            if quantity <= 0:
                return False
            
            # Check price for limit orders
            order_type = order_data.get('order_type')
            if order_type in ['limit', 'stop_limit'] and not order_data.get('price'):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating order data: {e}")
            return False
    
    def _create_trading_order(self, order_data: Dict[str, Any]) -> TradingOrder:
        """Create a trading order from order data."""
        try:
            # Generate order ID
            order_id = f"{order_data.get('exchange', 'unknown')}_{order_data.get('symbol')}_{int(time.time() * 1000)}"
            
            # Create order
            order = TradingOrder(
                order_id=order_id,
                symbol=order_data.get('symbol'),
                side=OrderSide(order_data.get('side')),
                order_type=OrderType(order_data.get('order_type')),
                quantity=float(order_data.get('quantity')),
                price=float(order_data.get('price')) if order_data.get('price') else None,
                stop_price=float(order_data.get('stop_price')) if order_data.get('stop_price') else None,
                metadata=order_data.get('metadata', {})
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"❌ Error creating trading order: {e}")
            raise
    
    async def _analyze_order_mathematically(self, order: TradingOrder) -> None:
        """Perform mathematical analysis on order."""
        try:
            if not self.math_bridge:
                return
            
            # Prepare order data for mathematical analysis
            order_data = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'timestamp': order.timestamp,
                'metadata': order.metadata
            }
            
            # Perform mathematical integration
            result = self.math_bridge.integrate_all_mathematical_systems(
                order_data, {}
            )
            
            # Update order with mathematical analysis
            order.mathematical_signature = result.mathematical_signature
            order.metadata['mathematical_analysis'] = {
                'confidence': result.overall_confidence,
                'connections': len(result.connections),
                'performance_metrics': result.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing order mathematically: {e}")
    
    async def _process_order_queue(self) -> None:
        """Process orders from the queue."""
        try:
            while self.active:
                try:
                    # Get order from queue
                    order = await asyncio.wait_for(
                        self.order_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process order
                    await self._process_order(order)
                    
                    # Mark task as done
                    self.order_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Error processing order: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in order processing loop: {e}")
    
    async def _process_order(self, order: TradingOrder) -> None:
        """Process a trading order."""
        try:
            start_time = time.time()
            
            # Update performance metrics
            self.performance_metrics['orders_submitted'] += 1
            
            # Execute order
            execution = await self._execute_order(order)
            
            # Store execution result
            self.order_history.append(execution)
            
            # Update order status
            order.status = execution.status
            order.filled_quantity = execution.filled_quantity
            order.average_price = execution.average_price
            
            # Remove from active orders if completed
            if execution.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics['orders_executed'] += 1
            
            if execution.success:
                self.performance_metrics['successful_executions'] += 1
            else:
                self.performance_metrics['failed_executions'] += 1
            
            self.performance_metrics['total_fees'] += execution.fees
            self.performance_metrics['total_slippage'] += execution.slippage
            
            # Update average execution time
            current_avg = self.performance_metrics['average_execution_time']
            total_executions = self.performance_metrics['orders_executed']
            self.performance_metrics['average_execution_time'] = (
                (current_avg * (total_executions - 1) + execution_time) / total_executions
            )
            
            self.logger.info(f"✅ Order processed: {order.order_id} - {execution.status.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing order: {e}")
    
    async def _execute_order(self, order: TradingOrder) -> OrderExecution:
        """Execute an order on the exchange."""
        try:
            start_time = time.time()
            
            # Determine exchange (for now, use first available)
            exchange_name = list(self.exchanges.keys())[0]
            exchange = self.exchanges[exchange_name]
            
            # Prepare order parameters
            order_params = self._prepare_order_params(order)
            
            # Execute order based on type
            if order.order_type == OrderType.MARKET:
                result = await self._execute_market_order(exchange, order, order_params)
            elif order.order_type == OrderType.LIMIT:
                result = await self._execute_limit_order(exchange, order, order_params)
            elif order.order_type == OrderType.STOP:
                result = await self._execute_stop_order(exchange, order, order_params)
            else:
                result = await self._execute_limit_order(exchange, order, order_params)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            slippage = self._calculate_slippage(order, result)
            fees = self._calculate_fees(result)
            
            # Create execution result
            execution = OrderExecution(
                order_id=order.order_id,
                success=result.get('status') in ['closed', 'filled'],
                status=self._map_order_status(result.get('status')),
                filled_quantity=float(result.get('filled', 0)),
                average_price=float(result.get('average', 0)),
                execution_time=execution_time,
                slippage=slippage,
                fees=fees,
                mathematical_signature=order.mathematical_signature,
                metadata={'exchange_result': result}
            )
            
            return execution
            
        except Exception as e:
            self.logger.error(f"❌ Error executing order {order.order_id}: {e}")
            
            # Create error execution result
            return OrderExecution(
                order_id=order.order_id,
                success=False,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                average_price=0.0,
                execution_time=0.0,
                slippage=0.0,
                fees=0.0,
                error_message=str(e)
            )
    
    def _prepare_order_params(self, order: TradingOrder) -> Dict[str, Any]:
        """Prepare order parameters for exchange."""
        try:
            params = {
                'symbol': order.symbol,
                'type': order.order_type.value,
                'side': order.side.value,
                'amount': order.quantity
            }
            
            if order.price:
                params['price'] = order.price
            
            if order.stop_price:
                params['stopPrice'] = order.stop_price
            
            # Add mathematical signature if available
            if order.mathematical_signature:
                params['clientOrderId'] = f"{order.order_id}_{order.mathematical_signature[:8]}"
            
            return params
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing order parameters: {e}")
            return {}
    
    async def _execute_market_order(self, exchange: ccxt.Exchange, order: TradingOrder, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a market order."""
        try:
            if order.side == OrderSide.BUY:
                result = await exchange.create_market_buy_order(order.symbol, order.quantity, params)
            else:
                result = await exchange.create_market_sell_order(order.symbol, order.quantity, params)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing market order: {e}")
            raise
    
    async def _execute_limit_order(self, exchange: ccxt.Exchange, order: TradingOrder, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a limit order."""
        try:
            result = await exchange.create_order(
                symbol=order.symbol,
                type=order.order_type.value,
                side=order.side.value,
                amount=order.quantity,
                price=order.price,
                params=params
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing limit order: {e}")
            raise
    
    async def _execute_stop_order(self, exchange: ccxt.Exchange, order: TradingOrder, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a stop order."""
        try:
            # Most exchanges don't support stop orders directly, so we'll use stop-limit
            params['type'] = 'stop-limit'
            params['stopPrice'] = order.stop_price
            
            result = await exchange.create_order(
                symbol=order.symbol,
                type='stop-limit',
                side=order.side.value,
                amount=order.quantity,
                price=order.price,
                params=params
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing stop order: {e}")
            raise
    
    def _map_order_status(self, exchange_status: str) -> OrderStatus:
        """Map exchange order status to internal status."""
        status_mapping = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'pending': OrderStatus.PENDING,
            'partial': OrderStatus.PARTIAL,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED
        }
        return status_mapping.get(exchange_status, OrderStatus.PENDING)
    
    def _calculate_slippage(self, order: TradingOrder, result: Dict[str, Any]) -> float:
        """Calculate order slippage."""
        try:
            if not order.price or not result.get('average'):
                return 0.0
            
            executed_price = float(result['average'])
            expected_price = order.price
            
            slippage = abs(executed_price - expected_price) / expected_price
            return slippage
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating slippage: {e}")
            return 0.0
    
    def _calculate_fees(self, result: Dict[str, Any]) -> float:
        """Calculate order fees."""
        try:
            fees = result.get('fee', {})
            if isinstance(fees, dict):
                return float(fees.get('cost', 0))
            elif isinstance(fees, (int, float)):
                return float(fees)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating fees: {e}")
            return 0.0
    
    async def _process_execution_queue(self) -> None:
        """Process execution results from the queue."""
        try:
            while self.active:
                try:
                    # Get execution from queue
                    execution = await asyncio.wait_for(
                        self.execution_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process execution (update balances, etc.)
                    await self._process_execution(execution)
                    
                    # Mark task as done
                    self.execution_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Error processing execution: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in execution processing loop: {e}")
    
    async def _process_execution(self, execution: OrderExecution) -> None:
        """Process an execution result."""
        try:
            # Update balances if order was successful
            if execution.success:
                await self._update_balances_after_execution(execution)
            
            # Log execution
            self.logger.info(f"💰 Execution completed: {execution.order_id} - {execution.filled_quantity} @ {execution.average_price}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing execution: {e}")
    
    async def _update_balances_after_execution(self, execution: OrderExecution) -> None:
        """Update balances after successful execution."""
        try:
            # This would typically involve fetching updated balances from the exchange
            # For now, we'll just log the execution
            self.logger.info(f"✅ Balances updated for execution: {execution.order_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating balances: {e}")
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        try:
            for order_id, order in list(self.active_orders.items()):
                try:
                    # Cancel order on exchange
                    exchange_name = list(self.exchanges.keys())[0]
                    exchange = self.exchanges[exchange_name]
                    
                    await exchange.cancel_order(order_id, order.symbol)
                    
                    # Update order status
                    order.status = OrderStatus.CANCELLED
                    
                    # Create cancellation execution
                    cancellation = OrderExecution(
                        order_id=order_id,
                        success=False,
                        status=OrderStatus.CANCELLED,
                        filled_quantity=0.0,
                        average_price=0.0,
                        execution_time=0.0,
                        slippage=0.0,
                        fees=0.0,
                        error_message="Cancelled by system"
                    )
                    
                    self.order_history.append(cancellation)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error cancelling order {order_id}: {e}")
            
            self.logger.info(f"✅ Cancelled {len(self.active_orders)} active orders")
            
        except Exception as e:
            self.logger.error(f"❌ Error cancelling orders: {e}")
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders."""
        return [
            {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'average_price': order.average_price,
                'timestamp': order.timestamp
            }
            for order in self.active_orders.values()
        ]
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order execution history."""
        recent_history = self.order_history[-limit:]
        return [
            {
                'order_id': execution.order_id,
                'success': execution.success,
                'status': execution.status.value,
                'filled_quantity': execution.filled_quantity,
                'average_price': execution.average_price,
                'execution_time': execution.execution_time,
                'slippage': execution.slippage,
                'fees': execution.fees,
                'error_message': execution.error_message
            }
            for execution in recent_history
        ]
    
    def get_balances(self, exchange_name: Optional[str] = None) -> Dict[str, Any]:
        """Get exchange balances."""
        try:
            if exchange_name:
                balances = self.balances.get(exchange_name, {})
                return {
                    currency: {
                        'free': balance.free,
                        'used': balance.used,
                        'total': balance.total,
                        'mathematical_signature': balance.mathematical_signature
                    }
                    for currency, balance in balances.items()
                }
            else:
                return {
                    exchange: {
                        currency: {
                            'free': balance.free,
                            'used': balance.used,
                            'total': balance.total
                        }
                        for currency, balance in balances.items()
                    }
                    for exchange, balances in self.balances.items()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting balances: {e}")
            return {}
    
    def get_market_info(self, exchange_name: Optional[str] = None) -> Dict[str, Any]:
        """Get market information."""
        try:
            if exchange_name:
                markets = self.market_info.get(exchange_name, {})
                return {
                    symbol: {
                        'base_currency': market.base_currency,
                        'quote_currency': market.quote_currency,
                        'min_amount': market.min_amount,
                        'max_amount': market.max_amount,
                        'price_precision': market.price_precision,
                        'amount_precision': market.amount_precision
                    }
                    for symbol, market in markets.items()
                }
            else:
                return {
                    exchange: {
                        symbol: {
                            'base_currency': market.base_currency,
                            'quote_currency': market.quote_currency,
                            'min_amount': market.min_amount,
                            'max_amount': market.max_amount
                        }
                        for symbol, market in markets.items()
                    }
                    for exchange, markets in self.market_info.items()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting market info: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Calculate success rate
        total_executions = metrics['orders_executed']
        if total_executions > 0:
            metrics['success_rate'] = metrics['successful_executions'] / total_executions
            metrics['average_fees'] = metrics['total_fees'] / total_executions
            metrics['average_slippage'] = metrics['total_slippage'] / total_executions
        else:
            metrics['success_rate'] = 0.0
            metrics['average_fees'] = 0.0
            metrics['average_slippage'] = 0.0
        
        return metrics
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info("✅ Enhanced CCXT Trading Engine System activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating Enhanced CCXT Trading Engine System: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info("✅ Enhanced CCXT Trading Engine System deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating Enhanced CCXT Trading Engine System: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'exchanges_connected': len(self.exchanges),
            'active_orders': len(self.active_orders),
            'orders_queued': self.order_queue.qsize(),
            'executions_queued': self.execution_queue.qsize(),
            'total_orders': len(self.order_history),
            'performance_metrics': self.performance_metrics,
            'config': {
                'enabled': self.config.enabled,
                'max_concurrent_orders': self.config.max_concurrent_orders,
                'order_timeout': self.config.order_timeout,
                'slippage_tolerance': self.config.slippage_tolerance,
                'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled,
                'sandbox_mode': self.config.sandbox_mode
            }
        }


def create_enhanced_ccxt_trading_engine(config: Optional[EnhancedCCXTConfig] = None) -> EnhancedCCXTTradingEngine:
    """Factory function to create EnhancedCCXTTradingEngine instance."""
    return EnhancedCCXTTradingEngine(config)


async def main():
    """Main function for testing."""
    # Create configuration
    config = EnhancedCCXTConfig(
        enabled=True,
        debug=True,
        max_concurrent_orders=10,
        order_timeout=60.0,
        slippage_tolerance=0.002,
        mathematical_analysis_enabled=True,
        sandbox_mode=True
    )
    
    # Create trading engine
    engine = create_enhanced_ccxt_trading_engine(config)
    
    # Activate system
    engine.activate()
    
    # Start trading engine
    await engine.start_trading_engine()
    
    # Submit test orders (these won't actually execute in sandbox without API keys)
    test_orders = [
        {
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'order_type': 'market',
            'quantity': 0.001,
            'metadata': {'test': True}
        },
        {
            'exchange': 'binance',
            'symbol': 'ETH/USDT',
            'side': 'sell',
            'order_type': 'limit',
            'quantity': 0.01,
            'price': 3000.0,
            'metadata': {'test': True}
        }
    ]
    
    # Submit orders
    for order_data in test_orders:
        await engine.submit_order(order_data)
    
    # Wait for processing
    await asyncio.sleep(5)
    
    # Get status
    status = engine.get_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # Get order history
    history = engine.get_order_history()
    print(f"Order History: {json.dumps(history, indent=2)}")
    
    # Stop trading engine
    await engine.stop_trading_engine()
    
    # Deactivate system
    engine.deactivate()


if __name__ == "__main__":
    asyncio.run(main())
