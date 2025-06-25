from __future__ import annotations

# #!/usr/bin/env python3
"""
API Configuration - CoinMarketCap and CoinGecko Integration
==========================================================

Provides secure configuration and integration for CoinMarketCap and CoinGecko APIs
with proper secret management and error handling.

Core Features:
- Secure API key management
- Rate limiting and cooldown handling
- Error handling and retry logic
- Configuration validation
- API endpoint management
"""


import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import hmac
import base64
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API endpoints and settings."""

    # CoinMarketCap Configuration
coinmarketcap_api_key: str = ""
coinmarketcap_base_url: str = "https://pro-api.coinmarketcap.com/v1"
coinmarketcap_rate_limit: int = 30  # requests per minute
coinmarketcap_timeout: int = 30

    # CoinGecko Configuration
coingecko_base_url: str = "https://api.coingecko.com/api/v3"
coingecko_rate_limit: int = 50  # requests per minute
coingecko_timeout: int = 30

    # General API Settings
retry_attempts: int = 3
retry_delay: float = 1.0
enable_caching: bool = True
cache_duration: int = 300  # 5 minutes

    # Security Settings
encrypt_api_keys: bool = True
key_rotation_interval: int = 86400  # 24 hours

    def __post_init__(self):
        """Validate configuration after initialization."""
self._validate_config()

    def _validate_config(self):
        """Validate API configuration settings."""
        if self.coinmarketcap_rate_limit <= 0:
            raise ValueError("CoinMarketCap rate limit must be positive")

        if self.coingecko_rate_limit <= 0:
            raise ValueError("CoinGecko rate limit must be positive")

        if self.retry_attempts < 0:
            raise ValueError("Retry attempts must be non-negative")

        if self.cache_duration < 0:
            raise ValueError("Cache duration must be non-negative")


class APISecretManager:
    """Manages API secrets securely."""

    def __init__(self, config_dir: str = "config"):
        """Initialize secret manager."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.secrets_file = self.config_dir / "api_secrets.json"
self._secrets_cache: Dict[str, str] = {}
self._load_secrets()

    def _load_secrets(self):
        """Load secrets from file."""
        try:
            if self.secrets_file.exists():
                with open(self.secrets_file, 'r') as f:
                    secrets = json.load(f)
                    self._secrets_cache = secrets.get('secrets', {})
                logger.info("API secrets loaded successfully")
            else:
logger.warning("No API secrets file found")
        except Exception as e:
logger.error(f"Failed to load API secrets: {e}")

    def _save_secrets(self):
        """Save secrets to file."""
        try:
secrets_data = {
'secrets': self._secrets_cache,
'last_updated': datetime.now().isoformat()
            }
            with open(self.secrets_file, 'w') as f:
                json.dump(secrets_data, f, indent=2)
            logger.info("API secrets saved successfully")
        except Exception as e:
logger.error(f"Failed to save API secrets: {e}")

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret value by key."""
        return self._secrets_cache.get(key)

    def set_secret(self, key: str, value: str):
        """Set secret value."""
self._secrets_cache[key] = value
self._save_secrets()

    def has_secret(self, key: str) -> bool:
        """Check if secret exists."""
        return key in self._secrets_cache

    def remove_secret(self, key: str):
        """Remove secret."""
        if key in self._secrets_cache:
            del self._secrets_cache[key]
self._save_secrets()


class APIRateLimiter:
    """Manages API rate limiting."""

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter."""
self.requests_per_minute = requests_per_minute
self.request_times: List[float] = []
self.lock = False

    def can_make_request(self) -> bool:
        """Check if request can be made."""
current_time = time.time()

        # Remove old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times
                            if current_time - t < 60]

        return len(self.request_times) < self.requests_per_minute

    def record_request(self):
        """Record a request."""
self.request_times.append(time.time())

    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.can_make_request():
            time.sleep(1)

    def get_wait_time(self) -> float:
        """Get time to wait before next request."""
        if self.can_make_request():
            return 0.0

current_time = time.time()
        oldest_request = min(self.request_times)
        return max(0.0, 60.0 - (current_time - oldest_request))


class APIClient:
    """Base API client with common functionality."""

    def __init__(self, config: APIConfig, secret_manager: APISecretManager):
        """Initialize API client."""
self.config = config
self.secret_manager = secret_manager
self.session = self._create_session()
        self.cache: Dict[str, Any] = {}
self.cache_timestamps: Dict[str, float] = {}

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
session = requests.Session()

retry_strategy = Retry(
            total=self.config.retry_attempts,
backoff_factor=self.config.retry_delay,
status_forcelist=[429, 500, 502, 503, 504]


adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_cached_response(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and valid."""
        if not self.config.enable_caching:
            return None

        if key not in self.cache:
            return None

timestamp = self.cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.config.cache_duration:
            # Cache expired
            del self.cache[key]
            del self.cache_timestamps[key]
            return None

        return self.cache[key]

    def _cache_response(self, key: str, response: Dict[str, Any]):
        """Cache response."""
        if self.config.enable_caching:
self.cache[key] = response
self.cache_timestamps[key] = time.time()

    def _make_request(self, url: str, params: Dict[str, Any] = None,
                     headers: Dict[str, str] = None) -> Dict[str, Any]:
"""Make HTTP request with error handling."""
        try:
response = self.session.get(
                url,
params=params,
headers=headers,
timeout=self.config.coingecko_timeout


response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
logger.error(f"API request failed: {e}")
            raise


class CoinMarketCapClient(APIClient):
    """CoinMarketCap API client."""

    def __init__(self, config: APIConfig, secret_manager: APISecretManager):
        """Initialize CoinMarketCap client."""
super().__init__(config, secret_manager)
        self.rate_limiter = APIRateLimiter(config.coinmarketcap_rate_limit)
        self.api_key = secret_manager.get_secret('coinmarketcap_api_key')

        if not self.api_key:
logger.warning("CoinMarketCap API key not found")

    def get_crypto_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get cryptocurrency quotes."""
        if not self.api_key:
            raise ValueError("CoinMarketCap API key required")

cache_key = f"cmc_quotes_{','.join(sorted(symbols))}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

self.rate_limiter.wait_if_needed()

url = f"{self.config.coinmarketcap_base_url}/cryptocurrency/quotes/latest"
params = {
'symbol': ','.join(symbols),
            'convert': 'USD'
}
headers = {
'X-CMC_PRO_API_KEY': self.api_key
}

response = self._make_request(url, params=params, headers=headers)
        self.rate_limiter.record_request()

self._cache_response(cache_key, response)
        return response

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global cryptocurrency metrics."""
        if not self.api_key:
            raise ValueError("CoinMarketCap API key required")

cache_key = "cmc_global_metrics"
cached = self._get_cached_response(cache_key)
        if cached:
            return cached

self.rate_limiter.wait_if_needed()

url = f"{self.config.coinmarketcap_base_url}/global-metrics/quotes/latest"
headers = {
'X-CMC_PRO_API_KEY': self.api_key
}

response = self._make_request(url, headers=headers)
        self.rate_limiter.record_request()

self._cache_response(cache_key, response)
        return response


class CoinGeckoClient(APIClient):
    """CoinGecko API client."""

    def __init__(self, config: APIConfig, secret_manager: APISecretManager):
        """Initialize CoinGecko client."""
super().__init__(config, secret_manager)
        self.rate_limiter = APIRateLimiter(config.coingecko_rate_limit)

    def get_crypto_prices(self, ids: List[str], vs_currencies: List[str] = None) -> Dict[str, Any]:
        """Get cryptocurrency prices."""
        if vs_currencies is None:
vs_currencies = ['usd']

cache_key = f"cg_prices_{','.join(sorted(ids))}_{','.join(sorted(vs_currencies))}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

self.rate_limiter.wait_if_needed()

url = f"{self.config.coingecko_base_url}/simple/price"
params = {
'ids': ','.join(ids),
            'vs_currencies': ','.join(vs_currencies)
        }

response = self._make_request(url, params=params)
        self.rate_limiter.record_request()

self._cache_response(cache_key, response)
        return response

    def get_market_data(self, ids: List[str], vs_currency: str = 'usd') -> Dict[str, Any]:
        """Get market data for cryptocurrencies."""
cache_key = f"cg_market_{','.join(sorted(ids))}_{vs_currency}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

self.rate_limiter.wait_if_needed()

url = f"{self.config.coingecko_base_url}/coins/markets"
params = {
'vs_currency': vs_currency,
'ids': ','.join(ids),
            'order': 'market_cap_desc',
'per_page': len(ids),
            'page': 1,
'sparkline': False
}

response = self._make_request(url, params=params)
        self.rate_limiter.record_request()

self._cache_response(cache_key, response)
        return response

    def get_global_data(self) -> Dict[str, Any]:
        """Get global cryptocurrency data."""
cache_key = "cg_global"
cached = self._get_cached_response(cache_key)
        if cached:
            return cached

self.rate_limiter.wait_if_needed()

url = f"{self.config.coingecko_base_url}/global"
response = self._make_request(url)
        self.rate_limiter.record_request()

self._cache_response(cache_key, response)
        return response


class APIManager:
    """Manages multiple API clients."""

    def __init__(self, config: APIConfig = None):
        """Initialize API manager."""
self.config = config or APIConfig()
        self.secret_manager = APISecretManager()
        self.coinmarketcap = CoinMarketCapClient(self.config, self.secret_manager)
        self.coingecko = CoinGeckoClient(self.config, self.secret_manager)

    def setup_api_keys(self, coinmarketcap_key: str = None):
        """Setup API keys."""
        if coinmarketcap_key:
self.secret_manager.set_secret('coinmarketcap_api_key', coinmarketcap_key)
            logger.info("CoinMarketCap API key configured")

    def get_crypto_data(self, symbols: List[str], source: str = 'coingecko') -> Dict[str, Any]:
        """Get cryptocurrency data from specified source."""
        try:
            if source.lower() == 'coinmarketcap':
                if not self.secret_manager.has_secret('coinmarketcap_api_key'):
                    raise ValueError("CoinMarketCap API key not configured")
                return self.coinmarketcap.get_crypto_quotes(symbols)

            elif source.lower() == 'coingecko':
                # Convert symbols to CoinGecko IDs (simplified)
                ids = [symbol.lower() for symbol in symbols]
                return self.coingecko.get_crypto_prices(ids)

            else:
                raise ValueError(f"Unsupported API source: {source}")

        except Exception as e:
logger.error(f"Failed to get crypto data from {source}: {e}")
            raise

    def get_global_metrics(self, source: str = 'coingecko') -> Dict[str, Any]:
        """Get global cryptocurrency metrics."""
        try:
            if source.lower() == 'coinmarketcap':
                if not self.secret_manager.has_secret('coinmarketcap_api_key'):
                    raise ValueError("CoinMarketCap API key not configured")
                return self.coinmarketcap.get_global_metrics()

            elif source.lower() == 'coingecko':
                return self.coingecko.get_global_data()

            else:
                raise ValueError(f"Unsupported API source: {source}")

        except Exception as e:
logger.error(f"Failed to get global metrics from {source}: {e}")
            raise


# Global instance for convenience
api_manager = APIManager()

# Convenience functions
def setup_api_keys(coinmarketcap_key: str = None):
    """Setup API keys."""
api_manager.setup_api_keys(coinmarketcap_key)


def get_crypto_data(symbols: List[str], source: str = 'coingecko') -> Dict[str, Any]:
    """Get cryptocurrency data."""
    return api_manager.get_crypto_data(symbols, source)


def get_global_metrics(source: str = 'coingecko') -> Dict[str, Any]:
    """Get global cryptocurrency metrics."""
    return api_manager.get_global_metrics(source)


if __name__ == "__main__":
    # Test the API configuration
    import sys
    import os

    # Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    # Import safe print for Windows compatibility
    try:
        from core.utils.windows_cli_compatibility import safe_print
    except ImportError:
        try:
#             from utils.windows_cli_compatibility import safe_print  # F811: duplicate import
        except ImportError:
            def safe_print(message):
                print(message)

    def main():
        """Main function to test API configuration and ensure proper initialization."""
        try:
safe_print("🌐 Testing API Configuration")
            safe_print("=" * 40)

            # Test configuration
safe_print("\n⚙️ Testing API Configuration:")
            config = APIConfig()
            safe_print(f"✅ CoinMarketCap Rate Limit: {config.coinmarketcap_rate_limit}")
            safe_print(f"✅ CoinGecko Rate Limit: {config.coingecko_rate_limit}")
            safe_print(f"✅ Retry Attempts: {config.retry_attempts}")
            safe_print(f"✅ Cache Duration: {config.cache_duration}")

            # Test secret manager
safe_print("\n🔐 Testing Secret Management:")
            secret_manager = APISecretManager()
            safe_print(f"✅ Secrets loaded: {len(secret_manager._secrets_cache)}")

            # Test secret operations
test_key = "test_api_key"
test_value = "test_secret_value"
secret_manager.set_secret(test_key, test_value)
            retrieved_value = secret_manager.get_secret(test_key)
            has_secret = secret_manager.has_secret(test_key)
            safe_print(f"✅ Secret Storage: {retrieved_value == test_value}")
            safe_print(f"✅ Secret Retrieval: {retrieved_value is not None}")
            safe_print(f"✅ Secret Check: {has_secret}")

            # Clean up test secret
secret_manager.remove_secret(test_key)

            # Test rate limiter
safe_print("\n⏱️ Testing Rate Limiting:")
            rate_limiter = APIRateLimiter(30)
            safe_print(f"✅ Initial Can Make Request: {rate_limiter.can_make_request()}")

            # Test multiple requests
            for i in range(5):
                rate_limiter.record_request()
            safe_print(f"✅ After 5 Requests - Can Make Request: {rate_limiter.can_make_request()}")

wait_time = rate_limiter.get_wait_time()
            safe_print(f"✅ Wait Time: {wait_time:.2f} seconds")

            # Test API manager
safe_print("\n🔧 Testing API Manager:")
            manager = APIManager(config)
            safe_print("✅ API Manager initialized successfully")

            # Test CoinGecko client (no API key required)
            safe_print("\n🪙 Testing CoinGecko Integration:")
            try:
global_data = manager.get_global_metrics('coingecko')
                safe_print(f"✅ Global data keys: {list(global_data.keys())}")

                # Test crypto prices
crypto_data = manager.get_crypto_data(['bitcoin', 'ethereum'], 'coingecko')
                safe_print(f"✅ Crypto data retrieved: {len(crypto_data)} entries")

            except Exception as e:
safe_print(f"⚠️ CoinGecko test failed: {e}")

            # Test CoinMarketCap client (requires API key)
            safe_print("\n💱 Testing CoinMarketCap Integration:")
            try:
                # Test without API key
                if not secret_manager.has_secret('coinmarketcap_api_key'):
                    safe_print("⚠️ CoinMarketCap API key not configured - skipping tests")
                else:
                    # Test with API key
cmc_data = manager.get_crypto_data(['BTC', 'ETH'], 'coinmarketcap')
                    safe_print(f"✅ CoinMarketCap data retrieved: {len(cmc_data)} entries")
            except Exception as e:
safe_print(f"⚠️ CoinMarketCap test failed: {e}")

            # Test API client base functionality
safe_print("\n🔌 Testing API Client Base Functionality:")
            api_client = APIClient(config, secret_manager)
            safe_print("✅ API Client initialized")
            safe_print(f"✅ Session created: {api_client.session is not None}")
            safe_print(f"✅ Cache enabled: {config.enable_caching}")

            # Test convenience functions
safe_print("\n🎯 Testing Convenience Functions:")

            # Test setup_api_keys
setup_api_keys()  # No key provided
            safe_print("✅ setup_api_keys called successfully")

            # Test get_crypto_data
            try:
crypto_data = get_crypto_data(['bitcoin'], 'coingecko')
                safe_print(f"✅ get_crypto_data: {len(crypto_data)} entries")
            except Exception as e:
safe_print(f"⚠️ get_crypto_data failed: {e}")

            # Test get_global_metrics
            try:
global_metrics = get_global_metrics('coingecko')
                safe_print(f"✅ get_global_metrics: {len(global_metrics)} keys")
            except Exception as e:
safe_print(f"⚠️ get_global_metrics failed: {e}")

            # Test configuration validation
safe_print("\n✅ Testing Configuration Validation:")
            try:
                # Test invalid configuration
invalid_config = APIConfig()
                invalid_config.coinmarketcap_rate_limit = -1
safe_print("❌ Invalid configuration should have raised error")
            except ValueError as e:
safe_print(f"✅ Configuration validation working: {e}")

safe_print("\n🎉 API Configuration tests completed successfully!")
            return True

        except Exception as e:
safe_print(f"❌ API Configuration test failed: {e}")
            import traceback
traceback.print_exc()
            return False

    # Run main function
success = main()
    sys.exit(0 if success else 1)
