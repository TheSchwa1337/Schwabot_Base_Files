# -*- coding: utf-8 -*-
""""""
Base exchange API implementation.

This module provides the base ExchangeAPI class that all exchange-specific
implementations inherit from.
""""""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..trading_models.containers import Balance, ExchangeConfig, MarketData, OrderRequest, OrderResponse
from ..trading_models.enums import DataType

logger = logging.getLogger(__name__)


class ExchangeAPI(ABC):
    """Base exchange API class."""

    def __init__(self, config: ExchangeConfig) -> None:
        """"""
        Initialize exchange API.

        Args:
            config: Exchange configuration.
        """"""
        self.config = config
        self.session = self._create_session()
        self.rate_limiter = None  # Will be set by subclasses

    def _create_session(self) -> requests.Session:
        """"""
        Create HTTP session with retry logic.

        Returns:
            Configured requests session.
        """"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry()
            total=self.config.retry_attempts,
                backoff_factor=self.config.retry_delay,
                    status_forcelist=[429, 500, 502, 503, 504],
                    )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _make_request()
        self,
            method: str,
                endpoint: str,
                params: Optional[Dict[str, Any]] = None,
                data: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None,
                signed: bool = False,
                ) -> Dict[str, Any]:
        """"""
        Make HTTP request to exchange API.

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
            headers = self._sign_request(method, endpoint, params, data, headers)

        # Make request
        try:
            response = self.session.request()
                method=method,
                    url=url,
                        params=params,
                        json=data,
                        headers=headers,
                        timeout=self.config.timeout,
                        )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    @abstractmethod
    def _sign_request()
        self,
            method: str,
                endpoint: str,
                params: Optional[Dict[str, Any]] = None,
                data: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None,
                ) -> Dict[str, str]:
        """"""
        Sign request for exchange-specific authentication.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters.
            data: Request data.
            headers: Request headers.

        Returns:
            Updated headers with signature.
        """"""
        pass

    @abstractmethod
    def get_ticker(self, symbol: str) -> MarketData:
        """"""
        Get ticker data for a given symbol.

        Args:
            symbol: The trading symbol (e.g., "BTC/USD").

        Returns:
            MarketData object containing ticker information.
        """"""
        pass

    @abstractmethod
    def get_order_book(self, symbol: str, level: int = 2) -> MarketData:
        """"""
        Get order book for a given symbol.

        Args:
            symbol: The trading symbol (e.g., "BTC/USD").
            level: The level of the order book depth (e.g., 1 for top, 2 for full).

        Returns:
            MarketData object containing order book information.
        """"""
        pass

    @abstractmethod
    def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """"""
        Place a new order on the exchange.

        Args:
            order_request: An OrderRequest object containing order details.

        Returns:
            An OrderResponse object with details of the placed order.
        """"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """"""
        Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            True if the order was successfully cancelled, False otherwise.
        """"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """"""
        Get the status of a specific order.

        Args:
            order_id: The ID of the order to retrieve status for.

        Returns:
            An OrderResponse object with the current status of the order.
        """"""
        pass

    @abstractmethod
    def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """"""
        Get account balance for a specific asset or all assets.

        Args:
            asset: Optional. The asset symbol (e.g., "BTC", "USD"). If None, returns all balances.

        Returns:
            A dictionary where keys are asset symbols and values are Balance objects.
        """"""
        pass

    @abstractmethod
    def get_klines()
        self,
            symbol: str,
                interval: str,
                start_time: Optional[int] = None,
                end_time: Optional[int] = None,
                limit: Optional[int] = None,
                ) -> MarketData:
        """"""
        Get candlestick data (klines) for a trading pair.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT").
            interval: Candlestick interval (e.g., "1m", "1h", "1d").
            start_time: Optional. Start timestamp in milliseconds.
            end_time: Optional. End timestamp in milliseconds.
            limit: Optional. Number of candlesticks to retrieve.

        Returns:
            MarketData object containing kline information.
        """"""
        pass

    @abstractmethod
    def get_exchange_info(self) -> Dict[str, Any]:
        """"""
        Get general exchange information, such as supported symbols, trading rules, etc.

        Returns:
            A dictionary containing exchange information.
        """"""
        pass

    @abstractmethod
    def get_server_time(self) -> int:
        """"""
        Get the current server time of the exchange.

        Returns:
            Server time in milliseconds.
        """"""
        pass

    @abstractmethod
    def get_all_tickers(self) -> List[MarketData]:
        """"""
        Get ticker data for all available symbols on the exchange.

        Returns:
            A list of MarketData objects, each containing ticker information for a symbol.
        """"""
        pass
