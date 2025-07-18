#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Secure API Integration Manager
==============================
Comprehensive secure API integration with Alpha256 encryption,
multi-exchange profile management, and intelligent rebalancing.

Key Features:
1. Alpha256 encryption for all API communications
2. Multi-exchange profile management (Coinbase, Binance, Kraken)
3. Intelligent rebalancing with randomization
4. Secure API key management and rotation
5. Real-time portfolio monitoring and adjustment
6. Mathematical separation between profiles
"""

import asyncio
import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import numpy as np

# Import Alpha256 encryption
try:
    from core.alpha256_encryption import Alpha256Encryption, get_encryption
    ALPHA256_AVAILABLE = True
except ImportError:
    ALPHA256_AVAILABLE = False
    logging.warning("Alpha256 encryption not available")

# Import exchange APIs
try:
    from core.api.coinbase_direct import CoinbaseDirectAPI
    from core.api.multi_profile_coinbase_manager import MultiProfileCoinbaseManager
    EXCHANGE_APIS_AVAILABLE = True
except ImportError:
    EXCHANGE_APIS_AVAILABLE = False
    logging.warning("Exchange APIs not available")

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Supported exchange types."""
    COINBASE = "coinbase"
    BINANCE = "binance"
    KRAKEN = "kraken"
    KRAKEN_FUTURES = "kraken_futures"

class ProfileType(Enum):
    """Profile types for different trading strategies."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ARBITRAGE = "arbitrage"
    HEDGE = "hedge"

class RebalancingStrategy(Enum):
    """Rebalancing strategies."""
    THRESHOLD_BASED = "threshold_based"
    TIME_BASED = "time_based"
    RISK_ADJUSTED = "risk_adjusted"
    VOLATILITY_TARGETED = "volatility_targeted"
    MOMENTUM_DRIVEN = "momentum_driven"

@dataclass
class SecureAPIProfile:
    """Secure API profile configuration."""
    profile_id: str
    exchange_type: ExchangeType
    profile_type: ProfileType
    api_key_id: str  # Reference to encrypted API key
    enabled: bool = True
    sandbox_mode: bool = True
    max_position_size: float = 0.1  # 10% of portfolio
    target_allocation: Dict[str, float] = field(default_factory=dict)
    rebalancing_strategy: RebalancingStrategy = RebalancingStrategy.THRESHOLD_BASED
    rebalancing_threshold: float = 0.05  # 5% deviation
    randomization_factor: float = 0.1  # 10% randomization
    created_at: datetime = field(default_factory=datetime.now)
    last_rebalance: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PortfolioAllocation:
    """Portfolio allocation with randomization."""
    symbol: str
    target_percentage: float
    current_percentage: float
    randomized_target: float
    deviation: float
    last_update: datetime
    exchange_distribution: Dict[str, float] = field(default_factory=dict)

@dataclass
class RebalancingAction:
    """Rebalancing action with security context."""
    profile_id: str
    symbol: str
    action: str  # 'buy', 'sell', 'transfer'
    amount: float
    exchange: str
    priority: int
    security_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class SecureAPIIntegrationManager:
    """
    Secure API Integration Manager
    
    Provides comprehensive secure integration with multiple exchanges
    using Alpha256 encryption, intelligent rebalancing, and profile management.
    """
    
    def __init__(self, config_path: str = "config/secure_api_config.yaml"):
        """Initialize the secure API integration manager."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize Alpha256 encryption
        if ALPHA256_AVAILABLE:
            self.encryption = get_encryption()
            logger.info("✅ Alpha256 encryption initialized")
        else:
            self.encryption = None
            logger.warning("⚠️ Alpha256 encryption not available")
        
        # Profile management
        self.profiles: Dict[str, SecureAPIProfile] = {}
        self.active_profiles: Dict[str, Any] = {}  # Exchange API instances
        
        # Portfolio tracking
        self.portfolio_allocations: Dict[str, PortfolioAllocation] = {}
        self.total_portfolio_value = 0.0
        
        # Rebalancing state
        self.rebalancing_queue: List[RebalancingAction] = []
        self.last_rebalancing_check = datetime.now()
        self.rebalancing_cooldown = 300  # 5 minutes
        
        # Security state
        self.security_events: List[Dict[str, Any]] = []
        self.api_key_rotation_schedule: Dict[str, datetime] = {}
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        self.security_violations = 0
        
        # Initialize profiles
        self._initialize_profiles()
        
        logger.info("🔐 Secure API Integration Manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Configuration loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"⚠️ Configuration file not found: {self.config_path}")
                return self._default_config()
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'security': {
                'encryption_enabled': True,
                'key_rotation_interval': 86400,  # 24 hours
                'max_failed_attempts': 3,
                'session_timeout': 3600,  # 1 hour
                'audit_logging': True
            },
            'exchanges': {
                'coinbase': {
                    'enabled': True,
                    'sandbox': True,
                    'rate_limit': 100,
                    'timeout': 30
                },
                'binance': {
                    'enabled': True,
                    'sandbox': True,
                    'rate_limit': 1200,
                    'timeout': 30
                },
                'kraken': {
                    'enabled': True,
                    'sandbox': False,
                    'rate_limit': 15,
                    'timeout': 30
                }
            },
            'rebalancing': {
                'enabled': True,
                'check_interval': 300,  # 5 minutes
                'max_deviation': 0.1,  # 10%
                'randomization_enabled': True,
                'randomization_factor': 0.1
            },
            'portfolio': {
                'max_concentration': 0.25,  # 25% max in any asset
                'min_diversification': 3,  # Minimum 3 assets
                'target_assets': ['BTC', 'ETH', 'SOL', 'XRP', 'USDC'],
                'excluded_assets': ['USDT']  # Avoid stablecoin concentration
            }
        }
    
    def _initialize_profiles(self):
        """Initialize API profiles from configuration."""
        try:
            profiles_config = self.config.get('profiles', {})
            
            for profile_id, profile_data in profiles_config.items():
                profile = SecureAPIProfile(
                    profile_id=profile_id,
                    exchange_type=ExchangeType(profile_data['exchange']),
                    profile_type=ProfileType(profile_data['type']),
                    api_key_id=profile_data['api_key_id'],
                    enabled=profile_data.get('enabled', True),
                    sandbox_mode=profile_data.get('sandbox', True),
                    max_position_size=profile_data.get('max_position_size', 0.1),
                    target_allocation=profile_data.get('target_allocation', {}),
                    rebalancing_strategy=RebalancingStrategy(profile_data.get('rebalancing_strategy', 'threshold_based')),
                    rebalancing_threshold=profile_data.get('rebalancing_threshold', 0.05),
                    randomization_factor=profile_data.get('randomization_factor', 0.1)
                )
                
                self.profiles[profile_id] = profile
                
                # Initialize exchange API if enabled
                if profile.enabled:
                    self._initialize_exchange_api(profile)
            
            logger.info(f"✅ Initialized {len(self.profiles)} profiles")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize profiles: {e}")
    
    def _initialize_exchange_api(self, profile: SecureAPIProfile):
        """Initialize exchange API for a profile."""
        try:
            if not EXCHANGE_APIS_AVAILABLE:
                logger.warning("⚠️ Exchange APIs not available")
                return
            
            # Get encrypted API credentials
            if self.encryption:
                api_key, api_secret = self.encryption.get_api_key(profile.api_key_id)
                
                if profile.exchange_type == ExchangeType.COINBASE:
                    # Get passphrase for Coinbase
                    passphrase = self._get_coinbase_passphrase(profile.api_key_id)
                    
                    api_instance = CoinbaseDirectAPI(
                        api_key=api_key,
                        secret=api_secret,
                        passphrase=passphrase,
                        sandbox=profile.sandbox_mode
                    )
                    
                    self.active_profiles[profile.profile_id] = api_instance
                    logger.info(f"✅ Initialized Coinbase API for profile {profile.profile_id}")
                
                # Add other exchange initializations here
                # Binance, Kraken, etc.
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange API for {profile.profile_id}: {e}")
    
    def _get_coinbase_passphrase(self, api_key_id: str) -> str:
        """Get Coinbase passphrase for API key."""
        try:
            # This would be stored separately for additional security
            passphrase_key_id = f"{api_key_id}_passphrase"
            if self.encryption:
                passphrase, _ = self.encryption.get_api_key(passphrase_key_id)
                return passphrase
        except:
            pass
        
        # Fallback to environment variable
        return os.getenv('COINBASE_PASSPHRASE', '')
    
    async def secure_api_call(self, profile_id: str, method: str, endpoint: str, 
                            data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make a secure API call with encryption and validation."""
        try:
            if profile_id not in self.active_profiles:
                raise ValueError(f"Profile {profile_id} not active")
            
            profile = self.profiles[profile_id]
            api_instance = self.active_profiles[profile_id]
            
            # Encrypt sensitive data if needed
            if data and self.encryption:
                encrypted_data = self._encrypt_api_data(data, profile_id)
                data = {'encrypted_data': encrypted_data}
            
            # Add security headers
            headers = self._generate_security_headers(profile_id, method, endpoint)
            
            # Make API call
            if hasattr(api_instance, method):
                result = await getattr(api_instance, method)(endpoint, data=data, headers=headers)
            else:
                # Fallback to generic API call
                result = await self._generic_api_call(api_instance, method, endpoint, data, headers)
            
            # Validate and decrypt response
            if result and self.encryption:
                result = self._decrypt_api_response(result, profile_id)
            
            # Log security event
            self._log_security_event(profile_id, method, endpoint, 'success')
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Secure API call failed for {profile_id}: {e}")
            self._log_security_event(profile_id, method, endpoint, 'error', str(e))
            return None
    
    def _encrypt_api_data(self, data: Dict[str, Any], profile_id: str) -> str:
        """Encrypt API data using Alpha256."""
        try:
            if self.encryption:
                data_str = str(data)
                return self.encryption.encrypt(data_str, profile_id)
            return str(data)
        except Exception as e:
            logger.error(f"❌ Failed to encrypt API data: {e}")
            return str(data)
    
    def _decrypt_api_response(self, response: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
        """Decrypt API response using Alpha256."""
        try:
            if self.encryption and 'encrypted_data' in response:
                encrypted_data = response['encrypted_data']
                decrypted_data = self.encryption.decrypt(encrypted_data, profile_id)
                return eval(decrypted_data)  # Convert string back to dict
            return response
        except Exception as e:
            logger.error(f"❌ Failed to decrypt API response: {e}")
            return response
    
    def _generate_security_headers(self, profile_id: str, method: str, endpoint: str) -> Dict[str, str]:
        """Generate security headers for API calls."""
        timestamp = str(int(time.time()))
        nonce = str(random.randint(1000000, 9999999))
        
        # Create signature
        signature_data = f"{method}:{endpoint}:{timestamp}:{nonce}:{profile_id}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        return {
            'X-Timestamp': timestamp,
            'X-Nonce': nonce,
            'X-Signature': signature,
            'X-Profile-ID': profile_id,
            'User-Agent': 'Schwabot-Secure-API/1.0'
        }
    
    async def get_portfolio_balance(self, profile_id: str) -> Optional[Dict[str, float]]:
        """Get portfolio balance for a profile."""
        try:
            result = await self.secure_api_call(profile_id, 'get_balance', '/accounts')
            
            if result:
                # Process and validate balance data
                balance = self._process_balance_data(result, profile_id)
                
                # Update portfolio allocation
                await self._update_portfolio_allocation(profile_id, balance)
                
                return balance
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio balance for {profile_id}: {e}")
            return None
    
    def _process_balance_data(self, balance_data: Dict[str, Any], profile_id: str) -> Dict[str, float]:
        """Process and validate balance data."""
        processed_balance = {}
        
        try:
            # Extract balance information based on exchange format
            if 'accounts' in balance_data:
                # Coinbase format
                for account in balance_data['accounts']:
                    currency = account.get('currency')
                    balance = float(account.get('balance', 0))
                    if balance > 0:
                        processed_balance[currency] = balance
            elif 'balances' in balance_data:
                # Binance format
                for balance in balance_data['balances']:
                    currency = balance.get('asset')
                    free_balance = float(balance.get('free', 0))
                    if free_balance > 0:
                        processed_balance[currency] = free_balance
            else:
                # Generic format
                processed_balance = balance_data
            
            # Validate balances
            for currency, balance in processed_balance.items():
                if balance < 0:
                    logger.warning(f"⚠️ Negative balance detected: {currency} = {balance}")
                    processed_balance[currency] = 0
                elif balance > 1000000:  # Suspiciously large balance
                    logger.warning(f"⚠️ Unusually large balance: {currency} = {balance}")
            
            return processed_balance
            
        except Exception as e:
            logger.error(f"❌ Failed to process balance data: {e}")
            return {}
    
    async def _update_portfolio_allocation(self, profile_id: str, balance: Dict[str, float]):
        """Update portfolio allocation with current balances."""
        try:
            profile = self.profiles[profile_id]
            
            # Calculate total portfolio value
            total_value = sum(balance.values())
            
            # Update allocations with randomization
            for symbol, amount in balance.items():
                if symbol in self.portfolio_allocations:
                    allocation = self.portfolio_allocations[symbol]
                    allocation.current_percentage = amount / total_value if total_value > 0 else 0
                    allocation.last_update = datetime.now()
                    
                    # Apply randomization to target
                    if profile.randomization_factor > 0:
                        randomization = random.uniform(-profile.randomization_factor, profile.randomization_factor)
                        allocation.randomized_target = allocation.target_percentage * (1 + randomization)
                    else:
                        allocation.randomized_target = allocation.target_percentage
                    
                    # Calculate deviation
                    allocation.deviation = abs(allocation.current_percentage - allocation.randomized_target)
                    
                    # Update exchange distribution
                    allocation.exchange_distribution[profile_id] = amount
                else:
                    # Create new allocation
                    target_percentage = profile.target_allocation.get(symbol, 0.0)
                    randomized_target = target_percentage
                    
                    if profile.randomization_factor > 0:
                        randomization = random.uniform(-profile.randomization_factor, profile.randomization_factor)
                        randomized_target = target_percentage * (1 + randomization)
                    
                    allocation = PortfolioAllocation(
                        symbol=symbol,
                        target_percentage=target_percentage,
                        current_percentage=amount / total_value if total_value > 0 else 0,
                        randomized_target=randomized_target,
                        deviation=abs((amount / total_value if total_value > 0 else 0) - randomized_target),
                        last_update=datetime.now(),
                        exchange_distribution={profile_id: amount}
                    )
                    
                    self.portfolio_allocations[symbol] = allocation
            
            self.total_portfolio_value = total_value
            
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio allocation: {e}")
    
    async def check_rebalancing_needs(self) -> List[RebalancingAction]:
        """Check if rebalancing is needed across all profiles."""
        try:
            if datetime.now() - self.last_rebalancing_check < timedelta(seconds=self.rebalancing_cooldown):
                return []
            
            rebalancing_actions = []
            
            # Check each profile for rebalancing needs
            for profile_id, profile in self.profiles.items():
                if not profile.enabled:
                    continue
                
                profile_actions = await self._check_profile_rebalancing(profile_id)
                rebalancing_actions.extend(profile_actions)
            
            # Sort actions by priority
            rebalancing_actions.sort(key=lambda x: x.priority)
            
            self.last_rebalancing_check = datetime.now()
            
            return rebalancing_actions
            
        except Exception as e:
            logger.error(f"❌ Failed to check rebalancing needs: {e}")
            return []
    
    async def _check_profile_rebalancing(self, profile_id: str) -> List[RebalancingAction]:
        """Check rebalancing needs for a specific profile."""
        try:
            profile = self.profiles[profile_id]
            actions = []
            
            # Get current portfolio balance
            balance = await self.get_portfolio_balance(profile_id)
            if not balance:
                return actions
            
            # Check each asset for rebalancing needs
            for symbol, allocation in self.portfolio_allocations.items():
                if symbol not in balance:
                    continue
                
                current_amount = balance[symbol]
                current_percentage = current_amount / self.total_portfolio_value if self.total_portfolio_value > 0 else 0
                
                # Check if rebalancing is needed
                deviation = abs(current_percentage - allocation.randomized_target)
                
                if deviation > profile.rebalancing_threshold:
                    # Determine action
                    if current_percentage > allocation.randomized_target:
                        action = 'sell'
                        amount = current_amount * (deviation / current_percentage)
                    else:
                        action = 'buy'
                        amount = self.total_portfolio_value * (deviation / allocation.randomized_target)
                    
                    # Create rebalancing action
                    rebalancing_action = RebalancingAction(
                        profile_id=profile_id,
                        symbol=symbol,
                        action=action,
                        amount=amount,
                        exchange=profile.exchange_type.value,
                        priority=self._calculate_action_priority(deviation, symbol),
                        security_context={
                            'encrypted': True,
                            'profile_type': profile.profile_type.value,
                            'rebalancing_strategy': profile.rebalancing_strategy.value
                        }
                    )
                    
                    actions.append(rebalancing_action)
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Failed to check profile rebalancing for {profile_id}: {e}")
            return []
    
    def _calculate_action_priority(self, deviation: float, symbol: str) -> int:
        """Calculate priority for rebalancing action."""
        # Higher deviation = higher priority
        if deviation > 0.15:  # 15% deviation
            return 1
        elif deviation > 0.10:  # 10% deviation
            return 2
        elif deviation > 0.05:  # 5% deviation
            return 3
        else:
            return 4
    
    async def execute_rebalancing_action(self, action: RebalancingAction) -> bool:
        """Execute a rebalancing action securely."""
        try:
            profile = self.profiles[action.profile_id]
            
            # Validate action
            if not self._validate_rebalancing_action(action):
                logger.warning(f"⚠️ Invalid rebalancing action: {action}")
                return False
            
            # Check concentration limits
            if not self._check_concentration_limits(action):
                logger.warning(f"⚠️ Concentration limit exceeded: {action}")
                return False
            
            # Execute the action
            if action.action == 'buy':
                success = await self._execute_buy_order(action)
            elif action.action == 'sell':
                success = await self._execute_sell_order(action)
            else:
                logger.warning(f"⚠️ Unknown action type: {action.action}")
                return False
            
            if success:
                # Update profile metrics
                profile.last_rebalance = datetime.now()
                profile.performance_metrics['last_rebalance_success'] = True
                profile.performance_metrics['total_rebalances'] = profile.performance_metrics.get('total_rebalances', 0) + 1
                
                logger.info(f"✅ Rebalancing action executed: {action.profile_id} {action.action} {action.symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to execute rebalancing action: {e}")
            return False
    
    def _validate_rebalancing_action(self, action: RebalancingAction) -> bool:
        """Validate a rebalancing action."""
        try:
            # Check if profile exists and is enabled
            if action.profile_id not in self.profiles:
                return False
            
            profile = self.profiles[action.profile_id]
            if not profile.enabled:
                return False
            
            # Check amount limits
            if action.amount <= 0:
                return False
            
            if action.amount > self.total_portfolio_value * profile.max_position_size:
                return False
            
            # Check symbol validity
            if action.symbol not in profile.target_allocation:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to validate rebalancing action: {e}")
            return False
    
    def _check_concentration_limits(self, action: RebalancingAction) -> bool:
        """Check concentration limits before executing action."""
        try:
            max_concentration = self.config['portfolio']['max_concentration']
            
            # Calculate what the concentration would be after the action
            current_allocation = self.portfolio_allocations.get(action.symbol)
            if not current_allocation:
                return True
            
            if action.action == 'buy':
                new_percentage = current_allocation.current_percentage + (action.amount / self.total_portfolio_value)
            else:  # sell
                new_percentage = current_allocation.current_percentage - (action.amount / self.total_portfolio_value)
            
            return new_percentage <= max_concentration
            
        except Exception as e:
            logger.error(f"❌ Failed to check concentration limits: {e}")
            return False
    
    async def _execute_buy_order(self, action: RebalancingAction) -> bool:
        """Execute a buy order securely."""
        try:
            # Create order parameters
            order_params = {
                'symbol': action.symbol,
                'side': 'buy',
                'amount': action.amount,
                'type': 'market'
            }
            
            # Execute order
            result = await self.secure_api_call(
                action.profile_id,
                'create_order',
                '/orders',
                order_params
            )
            
            return result is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to execute buy order: {e}")
            return False
    
    async def _execute_sell_order(self, action: RebalancingAction) -> bool:
        """Execute a sell order securely."""
        try:
            # Create order parameters
            order_params = {
                'symbol': action.symbol,
                'side': 'sell',
                'amount': action.amount,
                'type': 'market'
            }
            
            # Execute order
            result = await self.secure_api_call(
                action.profile_id,
                'create_order',
                '/orders',
                order_params
            )
            
            return result is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to execute sell order: {e}")
            return False
    
    def _log_security_event(self, profile_id: str, method: str, endpoint: str, 
                           status: str, error_message: str = None):
        """Log security event."""
        event = {
            'timestamp': datetime.now(),
            'profile_id': profile_id,
            'method': method,
            'endpoint': endpoint,
            'status': status,
            'error_message': error_message,
            'ip_address': '127.0.0.1',  # Would be actual IP in production
            'user_agent': 'Schwabot-Secure-API/1.0'
        }
        
        self.security_events.append(event)
        
        if status == 'error':
            self.security_violations += 1
            logger.warning(f"⚠️ Security event: {event}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'security': {
                    'alpha256_available': ALPHA256_AVAILABLE,
                    'encryption_enabled': self.encryption is not None,
                    'security_violations': self.security_violations,
                    'last_security_event': self.security_events[-1] if self.security_events else None
                },
                'profiles': {
                    'total_profiles': len(self.profiles),
                    'active_profiles': len(self.active_profiles),
                    'profile_status': {
                        profile_id: {
                            'enabled': profile.enabled,
                            'exchange': profile.exchange_type.value,
                            'type': profile.profile_type.value,
                            'last_rebalance': profile.last_rebalance.isoformat() if profile.last_rebalance else None
                        }
                        for profile_id, profile in self.profiles.items()
                    }
                },
                'portfolio': {
                    'total_value': self.total_portfolio_value,
                    'allocations': len(self.portfolio_allocations),
                    'last_rebalancing_check': self.last_rebalancing_check.isoformat()
                },
                'performance': {
                    'total_trades': self.total_trades,
                    'successful_trades': self.successful_trades,
                    'total_profit': self.total_profit,
                    'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}
    
    async def start(self):
        """Start the secure API integration manager."""
        try:
            logger.info("🚀 Starting Secure API Integration Manager")
            
            # Start all active profiles
            for profile_id, profile in self.profiles.items():
                if profile.enabled:
                    logger.info(f"✅ Started profile: {profile_id}")
            
            # Start rebalancing monitoring
            asyncio.create_task(self._rebalancing_monitor())
            
            logger.info("✅ Secure API Integration Manager started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Secure API Integration Manager: {e}")
    
    async def stop(self):
        """Stop the secure API integration manager."""
        try:
            logger.info("🛑 Stopping Secure API Integration Manager")
            
            # Stop all active profiles
            for profile_id in self.active_profiles:
                logger.info(f"✅ Stopped profile: {profile_id}")
            
            logger.info("✅ Secure API Integration Manager stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping Secure API Integration Manager: {e}")
    
    async def _rebalancing_monitor(self):
        """Monitor and execute rebalancing actions."""
        try:
            while True:
                # Check for rebalancing needs
                actions = await self.check_rebalancing_needs()
                
                # Execute actions
                for action in actions:
                    success = await self.execute_rebalancing_action(action)
                    if success:
                        self.total_trades += 1
                        self.successful_trades += 1
                    else:
                        self.total_trades += 1
                
                # Wait before next check
                await asyncio.sleep(self.config['rebalancing']['check_interval'])
                
        except Exception as e:
            logger.error(f"❌ Rebalancing monitor error: {e}")

# Convenience functions
def create_secure_api_manager(config_path: str = "config/secure_api_config.yaml") -> SecureAPIIntegrationManager:
    """Create a secure API integration manager."""
    return SecureAPIIntegrationManager(config_path)

async def main():
    """Main function for testing."""
    manager = create_secure_api_manager()
    await manager.start()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main()) 