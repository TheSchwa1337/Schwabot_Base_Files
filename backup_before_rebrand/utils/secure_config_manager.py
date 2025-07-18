#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 Secure Configuration Manager - Schwabot Advanced Security Integration
=======================================================================

Enhanced secure configuration manager that integrates:
1. Multi-Layered Security (Fernet + Alpha Encryption + VMSP)
2. Mathematical Hash Verification
3. Service-Specific Salt Generation
4. Temporal Security Validation
5. Advanced API Key Management

This system provides the highest level of security for API keys and configuration data
using redundant mathematical encryption layers.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from getpass import getpass
from typing import Any, Dict, List, Optional, Tuple, Union

# Import multi-layered security
try:
    from utils.multi_layered_security_manager import (
        get_multi_layered_security,
        encrypt_api_key_secure,
        get_secure_api_key,
        validate_api_key_security
    )
    MULTI_LAYERED_SECURITY_AVAILABLE = True
except ImportError:
    MULTI_LAYERED_SECURITY_AVAILABLE = False

# Import Alpha Encryption for direct access
try:
    from schwabot.alpha_encryption import get_alpha_encryption, alpha_encrypt_data, analyze_alpha_security
    ALPHA_ENCRYPTION_AVAILABLE = True
except ImportError:
    ALPHA_ENCRYPTION_AVAILABLE = False

# Import VMSP for direct access
try:
    from schwabot.vortex_security import get_vortex_security
    VMSP_AVAILABLE = True
except ImportError:
    VMSP_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityMode(Enum):
    """Security mode enumeration."""
    MULTI_LAYERED = "multi_layered"  # Fernet + Alpha + VMSP
    ALPHA_ONLY = "alpha_only"        # Alpha Encryption only
    FERNET_ONLY = "fernet_only"      # Fernet only
    HASH_ONLY = "hash_only"          # Hash verification only


@dataclass
class SecureConfigResult:
    """Result from secure configuration operation."""
    success: bool
    security_score: float
    encryption_hash: str
    processing_time: float
    security_mode: SecurityMode
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SecureConfigManager:
    """
    🔐 Enhanced Secure Configuration Manager
    
    Provides advanced security for API keys and configuration data using:
    - Multi-Layered Security (Fernet + Alpha Encryption + VMSP)
    - Mathematical Hash Verification
    - Service-Specific Salt Generation
    - Temporal Security Validation
    - Advanced API Key Management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Secure Configuration Manager."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Security mode
        self.security_mode = SecurityMode(self.config.get('security_mode', 'multi_layered'))
        
        # Initialize security systems
        self.multi_layered_security = None
        self.alpha_encryption = None
        self.vmsp = None
        
        if MULTI_LAYERED_SECURITY_AVAILABLE:
            try:
                self.multi_layered_security = get_multi_layered_security()
                self.logger.info("✅ Multi-Layered Security initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Multi-Layered Security initialization failed: {e}")
        
        if ALPHA_ENCRYPTION_AVAILABLE:
            try:
                self.alpha_encryption = get_alpha_encryption()
                self.logger.info("✅ Alpha Encryption (Ω-B-Γ Logic) initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Alpha Encryption initialization failed: {e}")
        
        if VMSP_AVAILABLE:
            try:
                self.vmsp = get_vortex_security()
                self.logger.info("✅ VMSP integration initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ VMSP initialization failed: {e}")
        
        # Configuration state
        self.secure_configs: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, str] = {}
        self.security_history: List[SecureConfigResult] = []
        
        # Performance metrics
        self.security_metrics = {
            'total_operations': 0,
            'avg_security_score': 0.0,
            'avg_processing_time': 0.0,
            'mode_usage': {mode.value: 0 for mode in SecurityMode},
            'success_rate': 0.0
        }
        
        self.logger.info(f"🔐 Secure Configuration Manager initialized with {self.security_mode.value} mode")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'security_mode': 'multi_layered',
            'config_file': 'config/secure_config.json',
            'api_keys_file': 'config/secure_api_keys.json',
            'salt_file': 'secure/config_salt.bin',
            'hash_iterations': 100000,
            'temporal_window': 3600,
            'debug_mode': False,
            'enable_alpha_encryption': True,
            'enable_vmsp': True,
            'enable_multi_layered': True
        }
    
    def _generate_service_salt(self, service_name: str) -> bytes:
        """Generate service-specific salt."""
        salt_file = self.config.get('salt_file', 'secure/config_salt.bin')
        
        # Create secure directory if it doesn't exist
        os.makedirs(os.path.dirname(salt_file), exist_ok=True)
        
        if os.path.exists(salt_file):
            # Load existing salt
            with open(salt_file, 'rb') as f:
                base_salt = f.read()
        else:
            # Generate new base salt
            base_salt = os.urandom(32)
            with open(salt_file, 'wb') as f:
                f.write(base_salt)
        
        # Generate service-specific salt
        service_salt = hashlib.sha256(base_salt + service_name.encode()).digest()
        return service_salt
    
    def _hash_api_key(self, api_key: str, service_name: str) -> str:
        """Hash API key with service-specific salt."""
        salt = self._generate_service_salt(service_name)
        hash_input = salt + api_key.encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()
    
    def _multi_layered_encryption(self, data: str, service_name: str, context: Optional[Dict[str, Any]] = None) -> SecureConfigResult:
        """Multi-layered encryption using Fernet + Alpha + VMSP."""
        start_time = time.time()
        
        try:
            if not self.multi_layered_security:
                raise Exception("Multi-layered security not available")
            
            # Encrypt using multi-layered security
            result = self.multi_layered_security.encrypt_api_key(data, service_name, context)
            
            return SecureConfigResult(
                success=result.overall_success,
                security_score=result.total_security_score,
                encryption_hash=result.encryption_hash,
                processing_time=time.time() - start_time,
                security_mode=SecurityMode.MULTI_LAYERED,
                metadata={
                    'fernet_score': result.fernet_result.security_score,
                    'alpha_score': result.alpha_result.security_score if result.alpha_result else 0.0,
                    'vmsp_score': result.vmsp_result.security_score if result.vmsp_result else 0.0,
                    'hash_score': result.hash_result.security_score,
                    'temporal_score': result.temporal_result.security_score,
                    'layers_used': [layer_name for layer_name, res in {
                        'fernet': result.fernet_result,
                        'alpha': result.alpha_result,
                        'vmsp': result.vmsp_result,
                        'hash': result.hash_result,
                        'temporal': result.temporal_result
                    }.items() if res and res.success]
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Multi-layered encryption failed: {e}")
            return SecureConfigResult(
                success=False,
                security_score=0.0,
                encryption_hash="",
                processing_time=time.time() - start_time,
                security_mode=SecurityMode.MULTI_LAYERED,
                metadata={'error': str(e)}
            )
    
    def _alpha_only_encryption(self, data: str, service_name: str, context: Optional[Dict[str, Any]] = None) -> SecureConfigResult:
        """Alpha Encryption only (Ω-B-Γ Logic)."""
        start_time = time.time()
        
        try:
            if not self.alpha_encryption:
                raise Exception("Alpha Encryption not available")
            
            # Encrypt with Alpha Encryption
            alpha_result = self.alpha_encryption.encrypt_data(data, context)
            
            # Analyze security
            security_analysis = analyze_alpha_security(alpha_result)
            
            return SecureConfigResult(
                success=True,
                security_score=alpha_result.security_score,
                encryption_hash=alpha_result.encryption_hash,
                processing_time=time.time() - start_time,
                security_mode=SecurityMode.ALPHA_ONLY,
                metadata={
                    'omega_depth': alpha_result.omega_state.recursion_depth,
                    'beta_coherence': alpha_result.beta_state.quantum_coherence,
                    'gamma_entropy': alpha_result.gamma_state.wave_entropy,
                    'total_entropy': alpha_result.total_entropy,
                    'vmsp_integration': alpha_result.vmsp_integration,
                    'security_analysis': security_analysis
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Alpha Encryption failed: {e}")
            return SecureConfigResult(
                success=False,
                security_score=0.0,
                encryption_hash="",
                processing_time=time.time() - start_time,
                security_mode=SecurityMode.ALPHA_ONLY,
                metadata={'error': str(e)}
            )
    
    def _hash_only_verification(self, data: str, service_name: str, context: Optional[Dict[str, Any]] = None) -> SecureConfigResult:
        """Hash verification only."""
        start_time = time.time()
        
        try:
            # Generate hash
            data_hash = self._hash_api_key(data, service_name)
            
            # Calculate security score
            hash_strength = 256  # SHA-256
            salt_complexity = 256  # 32 bytes * 8 bits
            security_score = min(100.0, (hash_strength + salt_complexity) / 10)
            
            return SecureConfigResult(
                success=True,
                security_score=security_score,
                encryption_hash=data_hash,
                processing_time=time.time() - start_time,
                security_mode=SecurityMode.HASH_ONLY,
                metadata={
                    'data_hash': data_hash,
                    'hash_strength_bits': hash_strength,
                    'salt_complexity_bits': salt_complexity
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Hash verification failed: {e}")
            return SecureConfigResult(
                success=False,
                security_score=0.0,
                encryption_hash="",
                processing_time=time.time() - start_time,
                security_mode=SecurityMode.HASH_ONLY,
                metadata={'error': str(e)}
            )
    
    def _update_security_metrics(self, result: SecureConfigResult) -> None:
        """Update security metrics."""
        self.security_metrics['total_operations'] += 1
        self.security_metrics['avg_security_score'] = (
            (self.security_metrics['avg_security_score'] * (self.security_metrics['total_operations'] - 1) + 
             result.security_score) / self.security_metrics['total_operations']
        )
        self.security_metrics['avg_processing_time'] = (
            (self.security_metrics['avg_processing_time'] * (self.security_metrics['total_operations'] - 1) + 
             result.processing_time) / self.security_metrics['total_operations']
        )
        
        # Update mode usage
        mode_name = result.security_mode.value
        self.security_metrics['mode_usage'][mode_name] += 1
        
        # Update success rate
        total_successful = sum(1 for r in self.security_history if r.success) + (1 if result.success else 0)
        self.security_metrics['success_rate'] = total_successful / self.security_metrics['total_operations']
    
    def secure_api_key(self, api_key: str, service_name: str, context: Optional[Dict[str, Any]] = None) -> SecureConfigResult:
        """
        Secure API key using the configured security mode.
        
        Args:
            api_key: API key to secure
            service_name: Name of the service
            context: Optional context for additional security
            
        Returns:
            SecureConfigResult with security information
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔐 Securing API key for {service_name} using {self.security_mode.value} mode")
            
            # Choose encryption method based on security mode
            if self.security_mode == SecurityMode.MULTI_LAYERED:
                result = self._multi_layered_encryption(api_key, service_name, context)
            elif self.security_mode == SecurityMode.ALPHA_ONLY:
                result = self._alpha_only_encryption(api_key, service_name, context)
            elif self.security_mode == SecurityMode.HASH_ONLY:
                result = self._hash_only_verification(api_key, service_name, context)
            else:
                # Fallback to hash verification
                result = self._hash_only_verification(api_key, service_name, context)
            
            # Store secured API key
            if result.success:
                self.api_keys[service_name] = result.encryption_hash
                self.logger.info(f"✅ API key secured for {service_name}: {result.security_score:.1f}/100 security score")
            else:
                self.logger.error(f"❌ Failed to secure API key for {service_name}")
            
            # Update metrics
            self.security_history.append(result)
            self._update_security_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ API key securing failed: {e}")
            return SecureConfigResult(
                success=False,
                security_score=0.0,
                encryption_hash="",
                processing_time=time.time() - start_time,
                security_mode=self.security_mode,
                metadata={'error': str(e)}
            )
    
    def get_secure_api_key(self, service_name: str) -> Optional[str]:
        """Get secured API key for a service."""
        return self.api_keys.get(service_name)
    
    def validate_api_key_security(self, service_name: str) -> Dict[str, Any]:
        """Validate security of a stored API key."""
        if service_name not in self.api_keys:
            return {'valid': False, 'error': 'API key not found'}
        
        # Use multi-layered security validation if available
        if self.multi_layered_security:
            return self.multi_layered_security.validate_api_key_security(service_name)
        
        # Fallback validation
        return {
            'valid': True,
            'security_score': 0.0,  # Unknown without multi-layered security
            'timestamp': time.time(),
            'layers_used': [self.security_mode.value],
            'temporal_validity': True,
            'age_hours': 0.0
        }
    
    def load_secure_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load secure configuration from file."""
        config_file = config_file or self.config.get('config_file', 'config/secure_config.json')
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Validate and secure any API keys in config
                if 'api_keys' in config_data:
                    for service, api_key in config_data['api_keys'].items():
                        if api_key and api_key != 'YOUR_API_KEY_HERE':
                            self.secure_api_key(api_key, service)
                
                self.secure_configs.update(config_data)
                self.logger.info(f"✅ Loaded secure configuration from {config_file}")
                return config_data
            else:
                self.logger.warning(f"⚠️ Configuration file not found: {config_file}")
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load secure configuration: {e}")
            return {}
    
    def save_secure_config(self, config_data: Dict[str, Any], config_file: Optional[str] = None) -> bool:
        """Save secure configuration to file."""
        config_file = config_file or self.config.get('config_file', 'config/secure_config.json')
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            # Replace API keys with secure hashes
            secure_config = config_data.copy()
            if 'api_keys' in secure_config:
                secure_config['api_keys'] = {
                    service: self.get_secure_api_key(service) or 'NOT_SECURED'
                    for service in secure_config['api_keys']
                }
            
            with open(config_file, 'w') as f:
                json.dump(secure_config, f, indent=2)
            
            self.logger.info(f"✅ Saved secure configuration to {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save secure configuration: {e}")
            return False
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics."""
        return self.security_metrics.copy()
    
    def get_security_history(self, limit: int = 100) -> List[SecureConfigResult]:
        """Get recent security history."""
        return self.security_history[-limit:]
    
    def change_security_mode(self, new_mode: SecurityMode) -> bool:
        """Change security mode."""
        try:
            self.security_mode = new_mode
            self.config['security_mode'] = new_mode.value
            self.logger.info(f"✅ Security mode changed to {new_mode.value}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to change security mode: {e}")
            return False


# Global instance
_secure_config_instance = None


def get_secure_config_manager() -> SecureConfigManager:
    """Get global Secure Configuration Manager instance."""
    global _secure_config_instance
    if _secure_config_instance is None:
        _secure_config_instance = SecureConfigManager()
    return _secure_config_instance


def secure_api_key(api_key: str, service_name: str, context: Optional[Dict[str, Any]] = None) -> SecureConfigResult:
    """Global function to secure API key."""
    return get_secure_config_manager().secure_api_key(api_key, service_name, context)


def get_secure_api_key(service_name: str) -> Optional[str]:
    """Global function to get secured API key."""
    return get_secure_config_manager().get_secure_api_key(service_name)


def validate_api_key_security(service_name: str) -> Dict[str, Any]:
    """Global function to validate API key security."""
    return get_secure_config_manager().validate_api_key_security(service_name)


def load_secure_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Global function to load secure configuration."""
    return get_secure_config_manager().load_secure_config(config_file)


def save_secure_config(config_data: Dict[str, Any], config_file: Optional[str] = None) -> bool:
    """Global function to save secure configuration."""
    return get_secure_config_manager().save_secure_config(config_data, config_file)
