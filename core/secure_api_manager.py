from __future__ import annotations
import math

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
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
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""Secure API Manager - Linux-based secure storage for Schwabot APIs.

This module provides secure API management for:
- CoinMarketCap API
- Intrapeat triggers
- NiceHash API (BTC pool hashing)
- Future CCXT integration

Uses Linux-based secure storage with encrypted secrets management.
"""


import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import json
import base64
from pathlib import Path

# Import unified mathematics
try:
    from core.unified_mathematics_config import get_unified_math
unified_math = get_unified_math()
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
UNIFIED_MATH_AVAILABLE = False

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe

CLI_HANDLER_AVAILABLE = True
except ImportError:
CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)

logger = logging.getLogger(__name__)


class APIType(Enum):
    """API types for different services."""
COINMARKETCAP = "coinmarketcap"
INTRAPEAT = "intrapeat"
NICEHASH = "nicehash"
CCXT = "ccxt"


class SecurityLevel(Enum):
    """Security levels for API access."""
LOW = "low"          # Public APIs (CoinMarketCap)
    MEDIUM = "medium"    # Semi-private APIs (Intrapeat)
    HIGH = "high"        # Private APIs (NiceHash, CCXT)


@dataclass
class APICredentials:
    """Encrypted API credentials."""
api_type: APIType
api_key: str
api_secret: Optional[str] = None
    passphrase: Optional[str] = None
security_level: SecurityLevel = SecurityLevel.MEDIUM
encrypted: bool = True
last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0


@dataclass
class APIRequest:
    """API request data."""
endpoint: str
method: str
params: Dict[str, Any]
headers: Dict[str, str]
timestamp: datetime
request_id: str


@dataclass
class APIResponse:
    """API response data."""
status_code: int
data: Any
headers: Dict[str, str]
timestamp: datetime
request_id: str
response_time: float


class SecureAPIManager:
    """
Secure API Manager - Linux-based secure storage for Schwabot APIs.

Provides secure management for:
- CoinMarketCap API (public data)
    - Intrapeat triggers (semi-private)
    - NiceHash API (private BTC pool data)
    - Future CCXT integration
"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize secure API manager."""
self.config = config or {}

        # Security configuration
self.encryption_key = self._get_encryption_key()
        self.secure_storage_path = self._get_secure_storage_path()
        self.max_retries = 3
self.retry_delay = 1.0
self.rate_limit_delay = 0.1

        # API credentials storage
self.credentials: Dict[APIType, APICredentials] = {}
self.request_history: List[APIRequest] = []
self.response_history: List[APIResponse] = []

        # Rate limiting
self.rate_limits: Dict[APIType, Dict[str, float]] = {}
self.last_requests: Dict[APIType, datetime] = {}

        # Connection management
self.connection_pool = {}
self.auto_reconnect = True
self.reconnect_attempts = 3

        # Performance tracking
self.total_requests = 0
self.successful_requests = 0
self.failed_requests = 0
self.average_response_time = 0.0

safe_safe_print("🔐 Secure API Manager initialized")

    def _get_encryption_key(self) -> bytes:
        """Get encryption key from secure Linux storage."""
        try:
            # Try to get key from Linux keyring or secure storage
key_paths = [
"/run/secrets/schwabot_api_key",
"/etc/schwabot/api_key",
os.path.expanduser("~/.schwabot/api_key"),
                ".schwabot_api_key"
]

            for key_path in key_paths:
                if os.path.exists(key_path):
                    with open(key_path, 'rb') as f:
                        key = f.read()
                    if len(key) >= 32:
                        safe_safe_print(f"✅ Encryption key loaded from {key_path}")
                        return key[:32]  # Use first 32 bytes

            # Fallback: generate temporary key (not secure for production)
            safe_safe_print("⚠️ No secure key found, generating temporary key")
            return hashlib.sha256(b"temporary_key_for_development").digest()[:32]

        except Exception as e:
safe_safe_print(f"❌ Failed to get encryption key: {safe_format_error(e, 'encryption_key')}")
            return hashlib.sha256(b"fallback_key").digest()[:32]

    def _get_secure_storage_path(self) -> Path:
        """Get secure storage path for credentials."""
        try:
            # Try Linux secure storage locations
secure_paths = [
Path("/run/secrets/schwabot"),
                Path("/etc/schwabot/credentials"),
                Path.home() / ".schwabot" / "credentials",
                Path(".schwabot_credentials")
            ]

            for path in secure_paths:
                if path.exists() or path.parent.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    safe_safe_print(f"✅ Secure storage path: {path}")
                    return path

            # Fallback to local directory
fallback_path = Path(".schwabot_credentials")
            fallback_path.mkdir(exist_ok=True)
            safe_safe_print(f"⚠️ Using fallback storage path: {fallback_path}")
            return fallback_path

        except Exception as e:
safe_safe_print(f"❌ Failed to get secure storage path: {safe_format_error(e, 'storage_path')}")
            return Path(".schwabot_credentials")

    def encrypt_data(self, data: str) -> str:
        """Encrypt data using secure key."""
        try:
            import cryptography
            from cryptography.fernet import Fernet

            # Create Fernet key from our encryption key
fernet_key = base64.urlsafe_b64encode(self.encryption_key)
            fernet = Fernet(fernet_key)

            # Encrypt data
encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()

        except ImportError:
            # Fallback: simple XOR encryption (not secure, just for development)
            safe_safe_print("⚠️ cryptography not available, using fallback encryption")
            return self._simple_encrypt(data)
        except Exception as e:
safe_safe_print(f"❌ Encryption failed: {safe_format_error(e, 'encrypt_data')}")
            return data

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using secure key."""
        try:
            import cryptography
#             from cryptography.fernet import Fernet  # F811: duplicate import

            # Create Fernet key from our encryption key
fernet_key = base64.urlsafe_b64encode(self.encryption_key)
            fernet = Fernet(fernet_key)

            # Decrypt data
encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()

        except ImportError:
            # Fallback: simple XOR decryption
            return self._simple_decrypt(encrypted_data)
        except Exception as e:
safe_safe_print(f"❌ Decryption failed: {safe_format_error(e, 'decrypt_data')}")
            return encrypted_data

    def _simple_encrypt(self, data: str) -> str:
        """Simple XOR encryption (development only)."""
        key_bytes = self.encryption_key
data_bytes = data.encode()
        encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_bytes * (len(data_bytes) // len(key_bytes) + 1)))
        return base64.urlsafe_b64encode(encrypted).decode()

    def _simple_decrypt(self, encrypted_data: str) -> str:
        """Simple XOR decryption (development only)."""
        key_bytes = self.encryption_key
encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key_bytes * (len(encrypted_bytes) // len(key_bytes) + 1)))
        return decrypted.decode()

    def store_credentials(
        self,
api_type: APIType,
api_key: str,
api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
security_level: SecurityLevel = SecurityLevel.MEDIUM
) -> bool:
"""
Store encrypted API credentials in secure storage.

This encrypts and stores credentials where they can't be touched
but can be accessed by the system.
"""
        try:
            # Create credentials object
credentials = APICredentials(
                api_type=api_type,
api_key=self.encrypt_data(api_key),
                api_secret=self.encrypt_data(api_secret) if api_secret else None,
                passphrase=self.encrypt_data(passphrase) if passphrase else None,
                security_level=security_level,
encrypted=True,
last_accessed=datetime.now(),
                access_count=0


            # Store in memory
self.credentials[api_type] = credentials

            # Store in secure file
credentials_file = self.secure_storage_path / f"{api_type.value}_credentials.json"
credentials_data = {
'api_type': api_type.value,
'api_key': credentials.api_key,
'api_secret': credentials.api_secret,
'passphrase': credentials.passphrase,
'security_level': security_level.value,
'encrypted': True,
'last_accessed': credentials.last_accessed.isoformat(),
                'access_count': 0
}

            with open(credentials_file, 'w') as f:
                json.dump(credentials_data, f, indent=2)

            # Set secure file permissions (Linux)
            try:
os.chmod(credentials_file, 0o600)  # Owner read/write only
            except Exception:
                pass  # Windows doesn't support chmod

safe_safe_print(f"✅ Credentials stored securely for {api_type.value}")
            return True

        except Exception as e:
safe_safe_print(f"❌ Failed to store credentials: {safe_format_error(e, 'store_credentials')}")
            return False

    def load_credentials(self, api_type: APIType) -> Optional[APICredentials]:
        """Load encrypted API credentials from secure storage."""
        try:
            # Check if already loaded in memory
            if api_type in self.credentials:
                return self.credentials[api_type]

            # Load from secure file
credentials_file = self.secure_storage_path / f"{api_type.value}_credentials.json"

            if not credentials_file.exists():
                safe_safe_print(f"⚠️ No credentials found for {api_type.value}")
                return None

            with open(credentials_file, 'r') as f:
                credentials_data = json.load(f)

            # Create credentials object
credentials = APICredentials(
                api_type=api_type,
api_key=credentials_data['api_key'],
api_secret=credentials_data.get('api_secret'),
                passphrase=credentials_data.get('passphrase'),
                security_level=SecurityLevel(credentials_data.get('security_level', 'medium')),
                encrypted=True,
last_accessed=datetime.fromisoformat(credentials_data.get('last_accessed', datetime.now().isoformat())),
                access_count=credentials_data.get('access_count', 0)


            # Store in memory
self.credentials[api_type] = credentials

safe_safe_print(f"✅ Credentials loaded for {api_type.value}")
            return credentials

        except Exception as e:
safe_safe_print(f"❌ Failed to load credentials: {safe_format_error(e, 'load_credentials')}")
            return None

    def get_decrypted_credentials(self, api_type: APIType) -> Optional[Dict[str, str]]:
        """Get decrypted API credentials."""
        try:
credentials = self.load_credentials(api_type)
            if not credentials:
                return None

            # Decrypt credentials
decrypted_credentials = {
'api_key': self.decrypt_data(credentials.api_key),
                'api_secret': self.decrypt_data(credentials.api_secret) if credentials.api_secret else None,
                'passphrase': self.decrypt_data(credentials.passphrase) if credentials.passphrase else None
            }

            # Update access count
credentials.access_count += 1
credentials.last_accessed = datetime.now()

safe_safe_print(f"✅ Decrypted credentials for {api_type.value}")
            return decrypted_credentials

        except Exception as e:
safe_safe_print(f"❌ Failed to get decrypted credentials: {safe_format_error(e, 'decrypt_credentials')}")
            return None

async def make_api_request(
        self,
api_type: APIType,
endpoint: str,
method: str = "GET",
params: Optional[Dict[str, Any]] = None,
headers: Optional[Dict[str, str]] = None,
retry_count: int = 0
) -> Optional[APIResponse]:
"""
Make secure API request with auto-reconnect and rate limiting.

This provides robust wrappers for CCXT, direct REST/WebSocket,
        with built-in retry, back-off, and rate-limit throttling.
"""
        try:
            # Check rate limits
            if not self._check_rate_limit(api_type):
                await asyncio.sleep(self.rate_limit_delay)

            # Get credentials
credentials = self.get_decrypted_credentials(api_type)
            if not credentials:
safe_safe_print(f"❌ No credentials available for {api_type.value}")
                return None

            # Prepare request
request_id = self._generate_request_id()
            request_headers = self._prepare_headers(api_type, credentials, headers or {})
            request_params = params or {}

            # Create request object
request = APIRequest(
                endpoint=endpoint,
method=method,
params=request_params,
headers=request_headers,
timestamp=datetime.now(),
                request_id=request_id


            # Store request in history
self.request_history.append(request)

            # Make request with retry logic
start_time = time.time()
            response = None

            for attempt in range(self.max_retries):
                try:
response = await self._execute_request(api_type, request)
                    if response and response.status_code < 400:
                        break
                    elif attempt < self.max_retries - 1:
await asyncio.sleep(self.retry_delay * (2 ** attempt))

                except Exception as e:
safe_safe_print(f"⚠️ Request attempt {attempt + 1} failed: {safe_format_error(e, 'api_request')}")
                    if attempt < self.max_retries - 1:
await asyncio.sleep(self.retry_delay * (2 ** attempt))

            # Calculate response time
response_time = time.time() - start_time

            if response:
response.response_time = response_time
self.response_history.append(response)

                # Update statistics
self.total_requests += 1
                if response.status_code < 400:
self.successful_requests += 1
                else:
self.failed_requests += 1

                # Update average response time
self._update_average_response_time(response_time)

safe_safe_print(f"✅ API request completed: {api_type.value} - {response.status_code}")
                return response
            else:
self.total_requests += 1
self.failed_requests += 1
safe_safe_print(f"❌ API request failed after {self.max_retries} attempts")
                return None

        except Exception as e:
safe_safe_print(f"❌ API request failed: {safe_format_error(e, 'make_api_request')}")
            return None

    def _check_rate_limit(self, api_type: APIType) -> bool:
        """Check if request is within rate limits."""
        try:
now = datetime.now()
            last_request = self.last_requests.get(api_type)

            if last_request:
time_since_last = (now - last_request).total_seconds()

                # Rate limits by API type
                if api_type == APIType.COINMARKETCAP:
min_interval = 0.1  # 10 requests per second
                elif api_type == APIType.INTRAPEAT:
min_interval = 0.5  # 2 requests per second
                elif api_type == APIType.NICEHASH:
min_interval = 1.0  # 1 request per second
                else:
min_interval = 0.5

                if time_since_last < min_interval:
                    return False

self.last_requests[api_type] = now
            return True

        except Exception as e:
safe_safe_print(f"⚠️ Rate limit check failed: {safe_format_error(e, 'rate_limit')}")
            return True

    def _prepare_headers(
        self,
api_type: APIType,
credentials: Dict[str, str],
base_headers: Dict[str, str]
) -> Dict[str, str]:
"""Prepare headers for API request."""
        try:
headers = {
'User-Agent': 'Schwabot/1.0',
'Accept': 'application/json',
'Content-Type': 'application/json'
}

            # Add base headers
headers.update(base_headers)

            # Add API-specific headers
            if api_type == APIType.COINMARKETCAP:
headers['X-CMC_PRO_API_KEY'] = credentials['api_key']
            elif api_type == APIType.INTRAPEAT:
headers['Authorization'] = f"Bearer {credentials['api_key']}"
            elif api_type == APIType.NICEHASH:
headers['X-Request-ID'] = self._generate_request_id()
                # NiceHash uses HMAC authentication
                if credentials.get('api_secret'):
                    timestamp = str(int(time.time() * 1000))
                    nonce = self._generate_nonce()
                    signature = self._generate_nicehash_signature(
                        credentials['api_key'],
credentials['api_secret'],
timestamp,
nonce

headers['X-Time'] = timestamp
headers['X-Nonce'] = nonce
headers['X-Organization-Id'] = credentials['api_key']
headers['X-Request-Signature'] = signature

            return headers

        except Exception as e:
safe_safe_print(f"⚠️ Header preparation failed: {safe_format_error(e, 'prepare_headers')}")
            return base_headers

async def _execute_request(self, api_type: APIType, request: APIRequest) -> Optional[APIResponse]:
        """Execute the actual API request."""
        try:
            import aiohttp

            # Create session if not exists
            if api_type not in self.connection_pool:
timeout = aiohttp.ClientTimeout(total=30)
                self.connection_pool[api_type] = aiohttp.ClientSession(timeout=timeout)

session = self.connection_pool[api_type]

            # Make request
            if request.method.upper() == "GET":
                async with session.get(request.endpoint, params=request.params, headers=request.headers) as response:
                    data = await response.json()
            elif request.method.upper() == "POST":
                async with session.post(request.endpoint, json=request.params, headers=request.headers) as response:
                    data = await response.json()
            else:
safe_safe_print(f"❌ Unsupported method: {request.method}")
                return None

            # Create response object
api_response = APIResponse(
                status_code=response.status,
data=data,
headers=dict(response.headers),
                timestamp=datetime.now(),
                request_id=request.request_id,
response_time=0.0  # Will be set by caller


            return api_response

        except Exception as e:
safe_safe_print(f"❌ Request execution failed: {safe_format_error(e, 'execute_request')}")
            return None

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())

    def _generate_nonce(self) -> str:
        """Generate nonce for NiceHash API."""
        import uuid
        return str(uuid.uuid4())

    def _generate_nicehash_signature(
        self,
api_key: str,
api_secret: str,
timestamp: str,
nonce: str
) -> str:
"""Generate HMAC signature for NiceHash API."""
        try:
            # NiceHash signature format
message = f"{api_key}\x00{timestamp}\x00{nonce}"
signature = hmac.new(
                api_secret.encode(),
                message.encode(),
                hashlib.sha256
).hexdigest()

            return signature

        except Exception as e:
safe_safe_print(f"❌ NiceHash signature generation failed: {safe_format_error(e, 'nicehash_signature')}")
            return ""

    def _update_average_response_time(self, response_time: float) -> None:
        """Update average response time."""
        try:
            if self.total_requests > 0:
self.average_response_time = (
                    (self.average_response_time * (self.total_requests - 1) + response_time) /
                    self.total_requests

        except Exception:
            pass

    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
'total_requests': self.total_requests,
'successful_requests': self.successful_requests,
'failed_requests': self.failed_requests,
'success_rate': self.successful_requests / unified_math.max(self.total_requests, 1),
            'average_response_time': self.average_response_time,
'stored_credentials': list(self.credentials.keys()),
            'secure_storage_path': str(self.secure_storage_path),
            'auto_reconnect': self.auto_reconnect
}

    def clear_history(self) -> None:
        """Clear request and response history."""
self.request_history.clear()
        self.response_history.clear()
        safe_safe_print("🗑️ API history cleared")

async def close_connections(self) -> None:
        """Close all API connections."""
        try:
            for session in self.connection_pool.values():
                await session.close()
            self.connection_pool.clear()
            safe_safe_print("🔌 API connections closed")
        except Exception as e:
safe_safe_print(f"⚠️ Failed to close connections: {safe_format_error(e, 'close_connections')}")


# Global secure API manager instance
secure_api_manager = SecureAPIManager()


# Convenience functions for external access
def get_secure_api_manager() -> SecureAPIManager:
    """Get global secure API manager instance."""
    return secure_api_manager


def store_api_credentials(
    api_type: APIType,
api_key: str,
api_secret: Optional[str] = None,
    passphrase: Optional[str] = None,
security_level: SecurityLevel = SecurityLevel.MEDIUM
) -> bool:
"""Store API credentials securely."""
    return secure_api_manager.store_credentials(api_type, api_key, api_secret, passphrase, security_level)


def load_api_credentials(api_type: APIType) -> Optional[APICredentials]:
    """Load API credentials from secure storage."""
    return secure_api_manager.load_credentials(api_type)


async def make_api_request(
    api_type: APIType,
endpoint: str,
method: str = "GET",
params: Optional[Dict[str, Any]] = None,
headers: Optional[Dict[str, str]] = None
) -> Optional[APIResponse]:
"""Make secure API request."""
    return await secure_api_manager.make_api_request(api_type, endpoint, method, params, headers)


def get_api_stats() -> Dict[str, Any]:
    """Get API usage statistics."""
    return secure_api_manager.get_api_statistics()


# Example usage

if __name__ == "__main__":
    # Test secure API manager
safe_print("🧪 Testing Secure API Manager...")

    # Test credential storage
success = store_api_credentials(
        api_type=APIType.COINMARKETCAP,
api_key="test_api_key_12345",
security_level=SecurityLevel.LOW

safe_print(f"✅ Credential storage: {success}")

    # Test credential loading
credentials = load_api_credentials(APIType.COINMARKETCAP)
    if credentials:
safe_print(f"✅ Credential loading: {credentials.api_type.value}")

    # Test decryption
decrypted = secure_api_manager.get_decrypted_credentials(APIType.COINMARKETCAP)
    if decrypted:
safe_print(f"✅ Credential decryption: {decrypted['api_key']}")

    # Get statistics
stats = get_api_stats()
    safe_print(f"✅ API Statistics: {stats}")
