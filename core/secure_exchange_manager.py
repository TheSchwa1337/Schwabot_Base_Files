#!/usr/bin/env python3
"""
Secure Exchange Manager - Professional API Key & Exchange Integration

Provides secure, properly labeled exchange integration with:
- Environment variable support for secrets
- Encrypted local storage fallback
- Clear distinction between public/private keys
- Validation and connectivity testing
- Comprehensive logging without exposing secrets
- Integration with automated trading pipeline

Security Features:
- Never logs or displays actual secret keys
- Validates keys before allowing trading
- Supports multiple exchanges with proper isolation
- Environment variable priority over local storage
- Encrypted local storage for development/testing

Usage:
    # Environment variables (recommended for production)
    export BINANCE_API_KEY="your_public_api_key"
    export BINANCE_API_SECRET="your_secret_key"
    
    # Or use secure storage
    exchange_manager = SecureExchangeManager()
    exchange_manager.setup_exchange("binance", api_key="...", secret="...")
"""

import os
import logging
import asyncio
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

# CCXT for exchange integration
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available. Install with: pip install ccxt")

# Local secure storage
try:
    from utils.secure_config_manager import SecureConfigManager
    SECURE_STORAGE_AVAILABLE = True
except ImportError:
    SECURE_STORAGE_AVAILABLE = False
    logging.warning("Secure storage not available. Using environment variables only.")

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Supported exchange types with proper labeling."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    OKX = "okx"

@dataclass
class ExchangeCredentials:
    """Securely stored exchange credentials with clear labeling."""
    exchange: ExchangeType
    api_key: str  # PUBLIC API KEY (can be logged safely)
    secret: str   # SECRET KEY (never logged)
    passphrase: Optional[str] = None  # Additional secret for some exchanges
    sandbox: bool = True
    testnet: bool = True
    
    def __post_init__(self):
        """Validate credentials after initialization."""
        if not self.api_key or not self.secret:
            raise ValueError(f"API key and secret are required for {self.exchange.value}")
        
        # Log only the public key (safe to display)
        logger.info(f"✅ Configured {self.exchange.value} with API key: {self.api_key[:8]}...")
        logger.info(f"🔐 Secret key configured (length: {len(self.secret)})")
        
        if self.passphrase:
            logger.info(f"🔐 Passphrase configured (length: {len(self.passphrase)})")

@dataclass
class ExchangeStatus:
    """Exchange connection and trading status."""
    exchange: ExchangeType
    connected: bool = False
    authenticated: bool = False
    trading_enabled: bool = False
    balance_available: bool = False
    last_check: Optional[float] = None
    error_message: Optional[str] = None
    
    def __str__(self) -> str:
        """Safe string representation without sensitive data."""
        status = "🟢" if self.connected else "🔴"
        return f"{status} {self.exchange.value}: Connected={self.connected}, Trading={self.trading_enabled}"

class SecureExchangeManager:
    """
    Secure exchange manager with proper key handling and validation.
    
    Security Features:
    - Environment variable priority
    - Encrypted local storage fallback
    - Never logs secrets
    - Validates connectivity before trading
    - Clear labeling of public vs private keys
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize secure exchange manager."""
        self.exchanges: Dict[ExchangeType, ExchangeCredentials] = {}
        self.status: Dict[ExchangeType, ExchangeStatus] = {}
        self.ccxt_instances: Dict[ExchangeType, Any] = {}
        
        # Initialize secure storage if available
        if SECURE_STORAGE_AVAILABLE:
            self.secure_config = SecureConfigManager()
        else:
            self.secure_config = None
            logger.warning("🔐 Secure storage not available. Using environment variables only.")
        
        # Load configuration
        self._load_exchange_configs()
        
        logger.info("🔐 Secure Exchange Manager initialized")
    
    def _load_exchange_configs(self):
        """Load exchange configurations from environment variables and secure storage."""
        logger.info("🔍 Loading exchange configurations...")
        
        for exchange in ExchangeType:
            try:
                # Try environment variables first (most secure)
                env_credentials = self._load_from_environment(exchange)
                if env_credentials:
                    self.exchanges[exchange] = env_credentials
                    self.status[exchange] = ExchangeStatus(exchange=exchange)
                    logger.info(f"✅ Loaded {exchange.value} from environment variables")
                    continue
                
                # Try secure storage as fallback
                if self.secure_config:
                    secure_credentials = self._load_from_secure_storage(exchange)
                    if secure_credentials:
                        self.exchanges[exchange] = secure_credentials
                        self.status[exchange] = ExchangeStatus(exchange=exchange)
                        logger.info(f"✅ Loaded {exchange.value} from secure storage")
                        continue
                
                logger.info(f"⚠️ No credentials found for {exchange.value}")
                
            except Exception as e:
                logger.error(f"❌ Error loading {exchange.value}: {e}")
    
    def _load_from_environment(self, exchange: ExchangeType) -> Optional[ExchangeCredentials]:
        """Load credentials from environment variables."""
        exchange_name = exchange.value.upper()
        
        # Environment variable naming convention
        api_key = os.environ.get(f"{exchange_name}_API_KEY")
        secret = os.environ.get(f"{exchange_name}_API_SECRET")
        passphrase = os.environ.get(f"{exchange_name}_PASSPHRASE")
        
        if api_key and secret:
            logger.info(f"🔍 Found {exchange.value} credentials in environment variables")
            return ExchangeCredentials(
                exchange=exchange,
                api_key=api_key,
                secret=secret,
                passphrase=passphrase,
                sandbox=True,  # Default to sandbox for safety
                testnet=True
            )
        
        return None
    
    def _load_from_secure_storage(self, exchange: ExchangeType) -> Optional[ExchangeCredentials]:
        """Load credentials from secure storage."""
        if not self.secure_config:
            return None
        
        try:
            exchange_name = exchange.value
            api_key = self.secure_config.get_api_key(f"{exchange_name}_api_key")
            secret = self.secure_config.get_api_key(f"{exchange_name}_secret")
            passphrase = self.secure_config.get_api_key(f"{exchange_name}_passphrase")
            
            if api_key and secret:
                logger.info(f"🔍 Found {exchange.value} credentials in secure storage")
                return ExchangeCredentials(
                    exchange=exchange,
                    api_key=api_key,
                    secret=secret,
                    passphrase=passphrase,
                    sandbox=True,
                    testnet=True
                )
        except Exception as e:
            logger.error(f"❌ Error loading from secure storage: {e}")
        
        return None
    
    def setup_exchange(self, exchange: ExchangeType, api_key: str, secret: str, 
                      passphrase: Optional[str] = None, sandbox: bool = True) -> bool:
        """
        Setup exchange credentials with proper validation.
        
        Args:
            exchange: Exchange type
            api_key: PUBLIC API KEY (safe to log)
            secret: SECRET KEY (never logged)
            passphrase: Additional secret for some exchanges
            sandbox: Use sandbox/testnet mode
        
        Returns:
            True if setup successful
        """
        try:
            logger.info(f"🔧 Setting up {exchange.value} exchange...")
            logger.info(f"📋 API Key: {api_key[:8]}... (public)")
            logger.info(f"🔐 Secret: [REDACTED] (length: {len(secret)})")
            
            if passphrase:
                logger.info(f"🔐 Passphrase: [REDACTED] (length: {len(passphrase)})")
            
            # Create credentials
            credentials = ExchangeCredentials(
                exchange=exchange,
                api_key=api_key,
                secret=secret,
                passphrase=passphrase,
                sandbox=sandbox,
                testnet=sandbox
            )
            
            # Store credentials
            self.exchanges[exchange] = credentials
            self.status[exchange] = ExchangeStatus(exchange=exchange)
            
            # Test connectivity
            if self._test_exchange_connection(exchange):
                logger.info(f"✅ {exchange.value} setup successful and connected")
                return True
            else:
                logger.error(f"❌ {exchange.value} setup failed - connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error setting up {exchange.value}: {e}")
            return False
    
    def _test_exchange_connection(self, exchange: ExchangeType) -> bool:
        """Test exchange connectivity without exposing secrets."""
        if not CCXT_AVAILABLE:
            logger.error("❌ CCXT not available for connection testing")
            return False
        
        try:
            credentials = self.exchanges.get(exchange)
            if not credentials:
                logger.error(f"❌ No credentials found for {exchange.value}")
                return False
            
            # Create CCXT instance
            exchange_class = getattr(ccxt, exchange.value)
            ccxt_instance = exchange_class({
                'apiKey': credentials.api_key,
                'secret': credentials.secret,
                'passphrase': credentials.passphrase,
                'sandbox': credentials.sandbox,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            # Test basic connectivity
            logger.info(f"🔍 Testing {exchange.value} connectivity...")
            
            # Fetch time (public endpoint, no authentication required)
            server_time = ccxt_instance.fetch_time()
            logger.info(f"✅ {exchange.value} server time: {server_time}")
            
            # Test authentication (private endpoint)
            try:
                balance = ccxt_instance.fetch_balance()
                logger.info(f"✅ {exchange.value} authentication successful")
                logger.info(f"💰 Balance available: {len(balance.get('total', {}))} currencies")
                
                self.status[exchange].connected = True
                self.status[exchange].authenticated = True
                self.status[exchange].balance_available = True
                self.status[exchange].trading_enabled = True
                self.ccxt_instances[exchange] = ccxt_instance
                
                return True
                
            except Exception as auth_error:
                logger.warning(f"⚠️ {exchange.value} authentication failed: {auth_error}")
                # Still mark as connected if we can reach the server
                self.status[exchange].connected = True
                self.status[exchange].authenticated = False
                self.ccxt_instances[exchange] = ccxt_instance
                return True
                
        except Exception as e:
            logger.error(f"❌ {exchange.value} connection test failed: {e}")
            self.status[exchange].error_message = str(e)
            return False
    
    def get_exchange_status(self) -> Dict[str, ExchangeStatus]:
        """Get status of all configured exchanges."""
        return {exchange.value: status for exchange, status in self.status.items()}
    
    def get_available_exchanges(self) -> List[ExchangeType]:
        """Get list of exchanges that are connected and ready for trading."""
        return [
            exchange for exchange, status in self.status.items()
            if status.connected and status.authenticated and status.trading_enabled
        ]
    
    def execute_trade(self, exchange: ExchangeType, symbol: str, side: str, 
                     amount: float, order_type: str = 'market') -> Dict[str, Any]:
        """
        Execute a trade with proper validation and logging.
        
        Args:
            exchange: Exchange to use
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order amount
            order_type: 'market' or 'limit'
        
        Returns:
            Order result or error information
        """
        try:
            # Validate exchange status
            if exchange not in self.status:
                return {"error": f"Exchange {exchange.value} not configured"}
            
            status = self.status[exchange]
            if not status.trading_enabled:
                return {"error": f"Trading not enabled for {exchange.value}"}
            
            # Get CCXT instance
            ccxt_instance = self.ccxt_instances.get(exchange)
            if not ccxt_instance:
                return {"error": f"Exchange {exchange.value} not connected"}
            
            # Log trade attempt (without sensitive data)
            logger.info(f"🎯 Executing {side} order: {amount} {symbol} on {exchange.value}")
            
            # Execute order
            if order_type == 'market':
                if side == 'buy':
                    order = ccxt_instance.create_market_buy_order(symbol, amount)
                else:
                    order = ccxt_instance.create_market_sell_order(symbol, amount)
            else:
                # For limit orders, you'd need price parameter
                return {"error": "Limit orders not implemented yet"}
            
            # Log successful order (without sensitive data)
            logger.info(f"✅ Order executed: {order.get('id', 'unknown')} - {order.get('status', 'unknown')}")
            
            return {
                "success": True,
                "order_id": order.get('id'),
                "status": order.get('status'),
                "symbol": order.get('symbol'),
                "side": order.get('side'),
                "amount": order.get('amount'),
                "filled": order.get('filled'),
                "remaining": order.get('remaining'),
                "cost": order.get('cost'),
                "fee": order.get('fee'),
            }
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {"error": str(e)}
    
    def get_balance(self, exchange: ExchangeType, currency: str = None) -> Dict[str, Any]:
        """Get account balance for specified exchange."""
        try:
            ccxt_instance = self.ccxt_instances.get(exchange)
            if not ccxt_instance:
                return {"error": f"Exchange {exchange.value} not connected"}
            
            balance = ccxt_instance.fetch_balance()
            
            if currency:
                return {
                    "currency": currency,
                    "free": balance.get('free', {}).get(currency, 0),
                    "used": balance.get('used', {}).get(currency, 0),
                    "total": balance.get('total', {}).get(currency, 0),
                }
            else:
                # Return all balances (filter out zero balances)
                total_balances = balance.get('total', {})
                non_zero = {curr: amount for curr, amount in total_balances.items() if amount > 0}
                return {"balances": non_zero}
                
        except Exception as e:
            logger.error(f"❌ Failed to get balance: {e}")
            return {"error": str(e)}
    
    def validate_trading_ready(self) -> Tuple[bool, List[str]]:
        """
        Validate that the system is ready for automated trading.
        
        Returns:
            (is_ready, list_of_issues)
        """
        issues = []
        
        if not CCXT_AVAILABLE:
            issues.append("CCXT library not available")
            return False, issues
        
        available_exchanges = self.get_available_exchanges()
        if not available_exchanges:
            issues.append("No exchanges configured and ready for trading")
            return False, issues
        
        # Check each available exchange
        for exchange in available_exchanges:
            status = self.status[exchange]
            if not status.connected:
                issues.append(f"{exchange.value} not connected")
            if not status.authenticated:
                issues.append(f"{exchange.value} not authenticated")
            if not status.trading_enabled:
                issues.append(f"{exchange.value} trading not enabled")
        
        is_ready = len(available_exchanges) > 0 and len(issues) == 0
        
        if is_ready:
            logger.info(f"✅ Trading system ready with {len(available_exchanges)} exchanges")
        else:
            logger.warning(f"⚠️ Trading system not ready: {issues}")
        
        return is_ready, issues
    
    def get_secure_summary(self) -> Dict[str, Any]:
        """Get a secure summary of exchange status without exposing secrets."""
        summary = {
            "total_exchanges": len(self.exchanges),
            "connected_exchanges": len([s for s in self.status.values() if s.connected]),
            "trading_ready": len(self.get_available_exchanges()),
            "exchanges": {}
        }
        
        for exchange, status in self.status.items():
            summary["exchanges"][exchange.value] = {
                "configured": exchange in self.exchanges,
                "connected": status.connected,
                "authenticated": status.authenticated,
                "trading_enabled": status.trading_enabled,
                "balance_available": status.balance_available,
                "last_check": status.last_check,
                "error": status.error_message
            }
        
        return summary


# Global instance for easy access
secure_exchange_manager = SecureExchangeManager()


def get_exchange_manager() -> SecureExchangeManager:
    """Get the global secure exchange manager instance."""
    return secure_exchange_manager


def setup_exchange_from_env(exchange_name: str) -> bool:
    """Setup exchange from environment variables."""
    try:
        exchange = ExchangeType(exchange_name.lower())
        manager = get_exchange_manager()
        
        # Check if already configured
        if exchange in manager.exchanges:
            logger.info(f"✅ {exchange.value} already configured")
            return True
        
        # Try to load from environment
        credentials = manager._load_from_environment(exchange)
        if credentials:
            manager.exchanges[exchange] = credentials
            manager.status[exchange] = ExchangeStatus(exchange=exchange)
            
            # Test connection
            if manager._test_exchange_connection(exchange):
                logger.info(f"✅ {exchange.value} setup from environment successful")
                return True
        
        logger.warning(f"⚠️ Could not setup {exchange.value} from environment variables")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error setting up {exchange_name}: {e}")
        return False


if __name__ == "__main__":
    # Test the secure exchange manager
    logging.basicConfig(level=logging.INFO)
    
    manager = SecureExchangeManager()
    
    print("\n🔐 SECURE EXCHANGE MANAGER TEST")
    print("=" * 40)
    
    # Show status
    status = manager.get_secure_summary()
    print(f"Total exchanges: {status['total_exchanges']}")
    print(f"Connected: {status['connected_exchanges']}")
    print(f"Trading ready: {status['trading_ready']}")
    
    # Show individual exchange status
    for exchange_name, exchange_status in status['exchanges'].items():
        print(f"\n{exchange_name}:")
        for key, value in exchange_status.items():
            print(f"  {key}: {value}")
    
    # Validate trading readiness
    is_ready, issues = manager.validate_trading_ready()
    print(f"\nTrading ready: {is_ready}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}") 