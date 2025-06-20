#!/usr/bin/env python3
"""Base exchange API implementation.

This module provides the base ExchangeAPI class that all exchange-specific
implementations inherit from.
"""

import hashlib
import hmac
import base64
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..trading_models.containers import (
    ExchangeConfig, OrderRequest, OrderResponse, MarketData, Balance
)
from ..trading_models.enums import DataType
from ..utils.cli_handler import CLIHandler, safe_log

logger = logging.getLogger(__name__)


class ExchangeAPI(ABC):
    """Base exchange API class."""
    
    def __init__(self, config: ExchangeConfig) -> None:
        """Initialize exchange API.
        
        Args:
            config: Exchange configuration.
        """
        self.config = config
        self.session = self._create_session()
        self.rate_limiter = None  # Will be set by subclasses
        
        # Initialize CLI compatibility
        self.cli_handler = CLIHandler()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic.
        
        Returns:
            Configured requests session.
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.retry_attempts,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def safe_print(self, message: str, force_ascii: Optional[bool] = None) -> None:
        """Safe print with CLI compatibility.
        
        Args:
            message: Message to print.
            force_ascii: Whether to force ASCII conversion.
        """
        if force_ascii is None:
            force_ascii = getattr(self.config, 'force_ascii_output', False)
        
        self.cli_handler.safe_print(message, force_ascii)
    
    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """Safe logging with CLI compatibility.
        
        Args:
            level: Log level.
            message: Log message.
            context: Additional context.
            
        Returns:
            True if logging was successful.
        """
        return safe_log(logger, level, message, context)
    
    def _make_request(self, method: str, endpoint: str, 
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request to exchange API.
        
        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters.
            data: Request data.
            headers: Request headers.
            signed: Whether request needs signature.
            
        Returns:
            API response data.
            
        Raises:
            Exception: If request fails.
        """
        url = f"{self.config.base_url}{endpoint}"
        
        # Prepare headers
        if headers is None:
            headers = {}
        
        # Add signature if required
        if signed:
            headers = self._sign_request(method, endpoint, params, data, headers)
        
        # Make request
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            self.safe_log('error', error_msg)
            raise Exception(error_msg) from e
    
    @abstractmethod
    def _sign_request(self, method: str, endpoint: str,
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Sign request for exchange-specific authentication.
        
        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters.
            data: Request data.
            headers: Request headers.
            
        Returns:
            Updated headers with signature.
        """
        pass
    
    def get_ticker(self, symbol: str) -> MarketData:
        """Get ticker data for symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Market data containing ticker information.
        """
        try:
            endpoint = f"/products/{symbol}/ticker"
            result = self._make_request("GET", endpoint)
            
            return MarketData(
                symbol=symbol,
                data_type=DataType.TICKER,
                timestamp=time.time(),
                data=result
            )
            
        except Exception as e:
            error_msg = f"Error getting ticker for {symbol}: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def get_order_book(self, symbol: str, level: int = 2) -> MarketData:
        """Get order book for symbol.
        
        Args:
            symbol: Trading symbol.
            level: Order book depth level.
            
        Returns:
            Market data containing order book information.
        """
        try:
            endpoint = f"/products/{symbol}/book"
            params = {"level": level}
            result = self._make_request("GET", endpoint, params=params)
            
            return MarketData(
                symbol=symbol,
                data_type=DataType.ORDER_BOOK,
                timestamp=time.time(),
                data=result
            )
            
        except Exception as e:
            error_msg = f"Error getting order book for {symbol}: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place order on exchange.
        
        Args:
            order_request: Order request details.
            
        Returns:
            Order response with execution details.
        """
        try:
            # Prepare order data
            order_data = {
                "product_id": order_request.symbol,
                "side": order_request.side.value,
                "type": order_request.order_type.value,
                "size": str(order_request.quantity)
            }
            
            if order_request.price:
                order_data["price"] = str(order_request.price)
            
            if order_request.client_order_id:
                order_data["client_order_id"] = order_request.client_order_id
            
            # Make API request
            endpoint = "/orders"
            result = self._make_request("POST", endpoint, data=order_data, signed=True)
            
            # Create order response
            order_response = OrderResponse(
                order_id=result.get('id', ''),
                client_order_id=result.get('client_order_id'),
                symbol=result.get('product_id', order_request.symbol),
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=float(result.get('size', order_request.quantity)),
                price=float(result.get('price', 0)) if result.get('price') else None,
                status=result.get('status', 'pending'),
                filled_quantity=float(result.get('filled_size', 0)),
                average_price=float(result.get('executed_value', 0)),
                commission=float(result.get('fill_fees', 0)),
                created_at=time.time(),
                metadata=result
            )
            
            return order_response
            
        except Exception as e:
            error_msg = f"Error placing order: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def get_balances(self) -> List[Balance]:
        """Get account balances.
        
        Returns:
            List of balance objects.
        """
        try:
            endpoint = "/accounts"
            result = self._make_request("GET", endpoint, signed=True)
            
            balances = []
            for account in result:
                balance = Balance(
                    currency=account.get('currency', ''),
                    available=float(account.get('available', 0)),
                    total=float(account.get('balance', 0)),
                    locked=float(account.get('hold', 0)),
                    timestamp=time.time()
                )
                balances.append(balance)
            
            return balances
            
        except Exception as e:
            error_msg = f"Error getting balances: {e}"
            self.safe_log('error', error_msg)
            raise 