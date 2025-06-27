# -*- coding: utf-8 -*-
# #!/usr/bin/env python3
"""Base exchange API implementation."""

This module provides the base ExchangeAPI class that all exchange-specific
implementations inherit from.
""""""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import safe print for Windows compatibility
try:
    from ..utils.cli_handler import safe_log, CLIHandler
    from ..trading_models.enums import DataType
    from ..trading_models.containers import ()
        OrderResponse, OrderRequest, MarketData,
        ExchangeConfig, Balance
    
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    # Fallback functions if imports fail
    def safe_print(message):
        print(message)

    def info(message):
        print(f"[INFO] {message}")

    def warn(message):
        print(f"[WARN] {message}")

    def error(message):
        print(f"[ERROR] {message}")

    def success(message):
        print(f"[SUCCESS] {message}")

    def debug(message):
        print(f"[DEBUG] {message}")

    # Fallback classes
    class Placeholder: pass
        def safe_print(self, message, force_ascii=False):
            print(message)

    class Placeholder: pass
        pass

class Placeholder: pass
        pass

class Placeholder: pass
        pass

class Placeholder: pass
        pass

class Placeholder: pass
        pass

class Placeholder: pass
            pass
logger = logging.getLogger(__name__)


class ExchangeAPI(ABC):
    """Base exchange API class."""

    def __init__(self: "ExchangeAPI", config: ExchangeConfig) -> None:
        """Initialize exchange API."""

        Args:
            config: Exchange configuration.
        """"""
        self.config = config
        self.session = self._create_session()
        self.rate_limiter = None  # Will be set by subclasses

        # Initialize CLI compatibility
        self.cli_handler = CLIHandler()

    def _create_session(self: "ExchangeAPI") -> requests.Session:
        """Create HTTP session with retry logic."""

        Returns:
            Configured requests session.
        """"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry()
            total=self.config.retry_attempts,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def safe_safe_print()
        self: "ExchangeAPI", message: str, force_ascii: Optional[bool] = None
     -> None:
        """Safe print with CLI compatibility."""

        Args:
            message: Message to print.
            force_ascii: Whether to force ASCII conversion.
        """"""
        if force_ascii is None:
            force_ascii = getattr(self.config, "force_ascii_output", False)

            self.cli_handler.safe_print(message, force_ascii)

    def safe_log()
        self: "ExchangeAPI", level: str, message: str, context: str = ""
     -> bool:
        """Safe logging with CLI compatibility."""

        Args:
            level: Log level.
            message: Log message.
            context: Additional context.

        Returns:
            True if logging was successful.
        """"""
        return safe_log(logger, level, message, context)

    def _make_request()
        self: "ExchangeAPI",
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        signed: bool = False,
     -> Dict[str, Any]:
        """Make HTTP request to exchange API."""

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
        """"""
        url = f"{self.config.base_url}{endpoint}"

        # Prepare headers
        if headers is None:
            headers = {}

        # Add signature if required
        if signed:
            headers.update(self._sign_request(method, endpoint, params, data))

        # Make request
        try:
            response = self.session.request()
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.config.timeout,
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error(f"\\u274c API request failed: {e}")
            raise

    @abstractmethod
    def _sign_request()
        self: "ExchangeAPI",
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
     -> Dict[str, str]:
        """Sign request for authentication."""

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters.
            data: Request data.

        Returns:
            Headers with signature.
        """"""
        pass

    @abstractmethod
    def get_balance(self: "ExchangeAPI") -> List[Balance]:
        """Get account balance."""

        Returns:
            List of balances.
        """"""
        pass

    @abstractmethod
    def get_market_data(self: "ExchangeAPI", symbol: str) -> MarketData:
        """Get market data for symbol."""

        Args:
            symbol: Trading symbol.

        Returns:
            Market data.
        """"""
        pass

    @abstractmethod
    def place_order(self: "ExchangeAPI", order: OrderRequest) -> OrderResponse:
        """Place order."""

        Args:
            order: Order request.

        Returns:
            Order response.
        """"""
        pass

    @abstractmethod
    def cancel_order(self: "ExchangeAPI", order_id: str) -> bool:
        """Cancel order."""

        Args:
            order_id: Order ID.

        Returns:
            True if cancelled successfully.
        """"""
        pass

    @abstractmethod
    def get_order_status(self: "ExchangeAPI", order_id: str) -> OrderResponse:
        """Get order status."""

        Args:
            order_id: Order ID.

        Returns:
            Order response.
        """"""
        pass

def _handle_rate_limit(self: "ExchangeAPI") -> None:
        """Handle rate limiting."""
        if self.rate_limiter:
        self.rate_limiter.wait_if_needed()

    def _validate_response(self: "ExchangeAPI",)
                           response: Dict[str, Any] -> bool:
        """Validate API response."""

        Args:
            response: API response.

        Returns:
            True if response is valid.
        """"""
        # Basic validation - subclasses can override
        return isinstance(response, dict) and "error" not in response

    def _log_request()
        self: "ExchangeAPI",
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
     -> None:
        """Log API request."""

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters.
        """"""
        debug(f"\\u1f517 {method} {endpoint}")
        if params:
            debug(f"   Params: {params}")

    def _log_response(self: "ExchangeAPI", response: Dict[str, Any]) -> None:
        """Log API response."""

        Args:
            response: API response.
        """"""
        debug(f"\\u1f4e5 Response: {len(str(response))} chars")

    def health_check(self: "ExchangeAPI") -> bool:
        """Check API health."""

        Returns:
            True if API is healthy.
        """"""
        try:
            # Simple health check - subclasses can override
            response = self._make_request("GET", "/health")
            return self._validate_response(response)
        except Exception as e:
            error(f"\\u274c Health check failed: {e}")
            return False

    def get_api_info(self: "ExchangeAPI") -> Dict[str, Any]:
        """Get API information."""

        Returns:
            API information.
        """"""
        return {}
            "base_url": self.config.base_url,
            "timeout": self.config.timeout,
            "retry_attempts": self.config.retry_attempts,
            "retry_delay": self.config.retry_delay,
        


# Module exports
__all__ = ["ExchangeAPI"]


