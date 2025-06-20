#!/usr/bin/env python3
"""Coinbase Pro/Advanced Trade API implementation.

This module provides Coinbase-specific API functionality with proper
authentication and error handling.
"""

import hashlib
import hmac
import base64
import json
import time
from typing import Any, Dict, Optional

from .base_api import ExchangeAPI
from ..trading_models.containers import ExchangeConfig
from ..utils.rate_limiter import RateLimiter


class CoinbaseAPI(ExchangeAPI):
    """Coinbase Pro/Advanced Trade API implementation."""
    
    def __init__(self, config: ExchangeConfig) -> None:
        """Initialize Coinbase API.
        
        Args:
            config: Exchange configuration.
        """
        # Set Coinbase-specific defaults
        if not config.base_url:
            if config.sandbox:
                config.base_url = "https://api-public.sandbox.exchange.coinbase.com"
            else:
                config.base_url = "https://api.exchange.coinbase.com"
        
        super().__init__(config)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(config.rate_limit, 60.0)
    
    def _sign_request(self, method: str, endpoint: str,
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Sign request for Coinbase API.
        
        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters.
            data: Request data.
            headers: Request headers.
            
        Returns:
            Updated headers with Coinbase signature.
        """
        try:
            timestamp = str(int(time.time()))
            
            # Create signature string
            signature_string = f"{timestamp}{method}{endpoint}"
            
            if data:
                signature_string += json.dumps(data, separators=(',', ':'))
            
            # Create signature
            signature = hmac.new(
                base64.b64decode(self.config.api_secret),
                signature_string.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            # Update headers
            if headers is None:
                headers = {}
            
            headers.update({
                'CB-ACCESS-KEY': self.config.api_key,
                'CB-ACCESS-SIGN': signature_b64,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'Content-Type': 'application/json'
            })
            
            return headers
            
        except Exception as e:
            error_msg = f"Error signing Coinbase request: {e}"
            self.safe_log('error', error_msg)
            raise 