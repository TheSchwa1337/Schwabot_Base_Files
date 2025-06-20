#!/usr/bin/env python3
"""Simplified BTC Integration - Bitcoin Trading Integration Layer.

This module provides a clean, simplified interface for Bitcoin trading
operations with mathematical optimization and comprehensive error handling.

The module is now restructured to use separate packages for:
- trading_models: Data containers and enums
- exchange_apis: Exchange-specific implementations
- utils: Utility classes and helpers

This eliminates flake8 issues by keeping each module focused and concise.
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional

from .trading_models.containers import (
    ExchangeConfig, OrderRequest, OrderResponse, MarketData, 
    Balance, PerformanceMetrics
)
from .trading_models.enums import ExchangeType
from .exchange_apis import ExchangeAPI, CoinbaseAPI
from .utils.cli_handler import CLIHandler, safe_log

logger = logging.getLogger(__name__)


class SimplifiedBTCIntegration:
    """Simplified Bitcoin trading integration system.
    
    This class provides a simplified interface for Bitcoin trading operations
    with mathematical optimization and comprehensive error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize simplified BTC integration.
        
        Args:
            config: Integration configuration.
        """
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Initialize CLI compatibility handler
        self.cli_handler = CLIHandler()
        
        # Exchange APIs
        self.exchanges: Dict[str, ExchangeAPI] = {}
        self.active_exchange: Optional[ExchangeAPI] = None
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            total_orders=0, successful_orders=0, failed_orders=0,
            average_execution_time=0.0, total_execution_time=0.0,
            average_slippage=0.0, total_volume=0.0, api_calls=0,
            api_errors=0, cache_hits=0, cache_misses=0
        )
        
        # Threading and synchronization
        self.integration_lock = threading.Lock()
        self.order_lock = threading.Lock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Initialize exchanges
        self._initialize_exchanges()
        
        # Start monitoring if enabled
        if self.config.get('enable_monitoring', True):
            self._start_monitoring()
        
        # Log initialization
        init_message = f"SimplifiedBTCIntegration v{self.version} initialized"
        safe_log(logger, 'info', init_message)
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default integration configuration.
        
        Returns:
            Default configuration dictionary.
        """
        return {
            'enable_monitoring': True,
            'enable_cache': True,
            'cache_timeout': 5.0,  # 5 seconds
            'max_retries': 3,
            'retry_delay': 1.0,
            'enable_cli_compatibility': True,
            'force_ascii_output': False,
            'enable_performance_tracking': True,
            'default_exchange': 'coinbase',
            'sandbox_mode': True,
            'rate_limit': 100,  # requests per minute
            'timeout': 30
        }
    
    def safe_print(self, message: str, 
                   force_ascii: Optional[bool] = None) -> None:
        """Safe print function with CLI compatibility.
        
        Args:
            message: Message to print.
            force_ascii: Whether to force ASCII conversion.
        """
        if force_ascii is None:
            force_ascii = self.config.get('force_ascii_output', False)
        
        self.cli_handler.safe_print(message, force_ascii)
    
    def safe_log(self, level: str, message: str, 
                 context: str = "") -> bool:
        """Safe logging with CLI compatibility.
        
        Args:
            level: Log level.
            message: Log message.
            context: Additional context.
            
        Returns:
            True if logging was successful.
        """
        return safe_log(logger, level, message, context)
    
    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        try:
            # Add default exchanges based on configuration
            default_exchange = self.config.get('default_exchange', 
                                               'coinbase')
            
            if default_exchange == 'coinbase':
                # Create Coinbase configuration
                coinbase_config = ExchangeConfig(
                    exchange_type=ExchangeType.COINBASE,
                    api_key=self.config.get('coinbase_api_key', ''),
                    api_secret=self.config.get('coinbase_api_secret', ''),
                    sandbox=self.config.get('sandbox_mode', True)
                )
                
                self.add_exchange(ExchangeType.COINBASE, coinbase_config)
                self.set_active_exchange('coinbase')
            
            self.safe_log('info', 
                          f"Initialized {len(self.exchanges)} exchanges")
            
        except Exception as e:
            error_msg = f"Error initializing exchanges: {e}"
            self.safe_log('error', error_msg)
    
    def add_exchange(self, exchange_type: ExchangeType, 
                     config: ExchangeConfig) -> bool:
        """Add exchange to the integration.
        
        Args:
            exchange_type: Type of exchange to add.
            config: Exchange configuration.
            
        Returns:
            True if exchange was added successfully.
        """
        try:
            with self.integration_lock:
                if exchange_type == ExchangeType.COINBASE:
                    exchange = CoinbaseAPI(config)
                    self.exchanges['coinbase'] = exchange
                    self.safe_log('info', 'Added Coinbase exchange')
                    return True
                else:
                    self.safe_log('warning', 
                                  f'Unsupported exchange type: '
                                  f'{exchange_type}')
                    return False
                    
        except Exception as e:
            error_msg = f"Error adding exchange {exchange_type}: {e}"
            self.safe_log('error', error_msg)
            return False
    
    def set_active_exchange(self, exchange_name: str) -> bool:
        """Set the active exchange for operations.
        
        Args:
            exchange_name: Name of exchange to set as active.
            
        Returns:
            True if exchange was set successfully.
        """
        try:
            with self.integration_lock:
                if exchange_name in self.exchanges:
                    self.active_exchange = self.exchanges[exchange_name]
                    self.safe_log('info', 
                                  f'Set active exchange: {exchange_name}')
                    return True
                else:
                    self.safe_log('warning', 
                                  f'Exchange not found: {exchange_name}')
                    return False
                    
        except Exception as e:
            error_msg = f"Error setting active exchange: {e}"
            self.safe_log('error', error_msg)
            return False
    
    def get_ticker(self, symbol: str, 
                   exchange_name: Optional[str] = None) -> MarketData:
        """Get ticker data for symbol.
        
        Args:
            symbol: Trading symbol.
            exchange_name: Optional exchange name override.
            
        Returns:
            Market data containing ticker information.
        """
        try:
            exchange = self._get_exchange(exchange_name)
            return exchange.get_ticker(symbol)
            
        except Exception as e:
            error_msg = f"Error getting ticker for {symbol}: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def get_order_book(self, symbol: str, level: int = 2,
                       exchange_name: Optional[str] = None) -> MarketData:
        """Get order book for symbol.
        
        Args:
            symbol: Trading symbol.
            level: Order book depth level.
            exchange_name: Optional exchange name override.
            
        Returns:
            Market data containing order book information.
        """
        try:
            exchange = self._get_exchange(exchange_name)
            return exchange.get_order_book(symbol, level)
            
        except Exception as e:
            error_msg = f"Error getting order book for {symbol}: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def place_order(self, order_request: OrderRequest,
                    exchange_name: Optional[str] = None) -> OrderResponse:
        """Place order on exchange.
        
        Args:
            order_request: Order request details.
            exchange_name: Optional exchange name override.
            
        Returns:
            Order response with execution details.
        """
        try:
            exchange = self._get_exchange(exchange_name)
            
            # Track performance
            start_time = time.time()
            
            # Place order
            order_response = exchange.place_order(order_request)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_order_metrics(order_response, execution_time)
            
            return order_response
            
        except Exception as e:
            error_msg = f"Error placing order: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def get_balances(self, 
                     exchange_name: Optional[str] = None) -> List[Balance]:
        """Get account balances.
        
        Args:
            exchange_name: Optional exchange name override.
            
        Returns:
            List of balance objects.
        """
        try:
            exchange = self._get_exchange(exchange_name)
            return exchange.get_balances()
            
        except Exception as e:
            error_msg = f"Error getting balances: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def _get_exchange(self, 
                      exchange_name: Optional[str] = None) -> ExchangeAPI:
        """Get exchange instance.
        
        Args:
            exchange_name: Optional exchange name override.
            
        Returns:
            Exchange API instance.
            
        Raises:
            ValueError: If no exchange is available.
        """
        if exchange_name:
            if exchange_name in self.exchanges:
                return self.exchanges[exchange_name]
            else:
                raise ValueError(f"Exchange not found: {exchange_name}")
        
        if self.active_exchange:
            return self.active_exchange
        
        raise ValueError("No active exchange available")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        if not self.config.get('enable_performance_tracking', True):
            return
        
        # This would update various performance metrics
        # Implementation depends on specific tracking requirements
        pass
    
    def _update_order_metrics(self, order_response: OrderResponse,
                             execution_time: float) -> None:
        """Update order-related performance metrics.
        
        Args:
            order_response: Order response from exchange.
            execution_time: Time taken to execute order.
        """
        if not self.config.get('enable_performance_tracking', True):
            return
        
        with self.integration_lock:
            self.performance_metrics.total_orders += 1
            self.performance_metrics.total_execution_time += execution_time
            
            if order_response.status.value in ['filled', 'partially_filled']:
                self.performance_metrics.successful_orders += 1
                self.performance_metrics.total_volume += (
                    order_response.filled_quantity)
            else:
                self.performance_metrics.failed_orders += 1
            
            # Update average execution time
            if self.performance_metrics.total_orders > 0:
                self.performance_metrics.average_execution_time = (
                    self.performance_metrics.total_execution_time / 
                    self.performance_metrics.total_orders
                )
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """Get performance metrics summary.
        
        Returns:
            Current performance metrics.
        """
        return self.performance_metrics
    
    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        self.safe_log('info', 'Started monitoring thread')
    
    def _monitoring_loop(self) -> None:
        """Monitor background tasks."""
        while self.monitoring_active:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for monitoring interval
                time.sleep(5.0)  # 5 second interval
                
            except Exception as e:
                error_msg = f"Error in monitoring loop: {e}"
                self.safe_log('error', error_msg)
                time.sleep(10.0)  # Longer sleep on error


def main() -> None:
    """Run main function for testing."""
    # Create integration instance
    integration = SimplifiedBTCIntegration()
    
    # Example usage
    try:
        # Get ticker data
        ticker = integration.get_ticker("BTC-USD")
        print(f"BTC Price: {ticker.data.get('price', 'N/A')}")
        
        # Get balances
        balances = integration.get_balances()
        for balance in balances:
            print(f"{balance.currency}: {balance.available}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 