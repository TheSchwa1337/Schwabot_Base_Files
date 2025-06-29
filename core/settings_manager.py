#!/usr/bin/env python3
"""
Schwabot Settings Manager
========================

Comprehensive configuration management system with:
- API credentials and settings
- Performance optimization settings
- Risk management parameters
- System configuration
- Persistent storage with encryption
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

# Cryptography import with fallback
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None

logger = logging.getLogger(__name__)


@dataclass
class APISettings:
    """API configuration settings."""
    coinbase_api_key: str = ""
    coinbase_secret: str = ""
    coinbase_passphrase: str = ""
    sandbox_mode: bool = True
    api_timeout: float = 30.0
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class PerformanceSettings:
    """Performance optimization settings."""
    gpu_enabled: bool = True
    cpu_threads: int = 4
    memory_limit_mb: int = 2048
    update_interval: float = 1.0
    batch_size: int = 100
    cache_size: int = 1000
    async_processing: bool = True


@dataclass
class RiskSettings:
    """Risk management settings."""
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.05    # 5% daily stop loss
    max_drawdown: float = 0.15      # 15% maximum drawdown
    profit_target: float = 0.20     # 20% profit target
    stop_loss_percent: float = 0.02 # 2% stop loss
    position_timeout_hours: int = 24
    emergency_stop_enabled: bool = True


@dataclass
class TradingSettings:
    """Trading strategy settings."""
    trading_mode: str = "demo"  # demo, live, backtest
    base_currency: str = "USD"
    target_currency: str = "BTC"
    min_trade_amount: float = 10.0
    max_concurrent_trades: int = 5
    rebalance_frequency_hours: int = 24
    signal_confidence_threshold: float = 0.7


@dataclass
class SystemSettings:
    """Complete system configuration."""
    api: APISettings
    performance: PerformanceSettings
    risk: RiskSettings
    trading: TradingSettings
    
    def __init__(self):
        self.api = APISettings()
        self.performance = PerformanceSettings()
        self.risk = RiskSettings()
        self.trading = TradingSettings()


class SettingsManager:
    """Manages system configuration with encryption and persistence."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or os.path.expanduser("~/.schwabot"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "config.yaml"
        self.encrypted_file = self.config_dir / "secure.enc"
        self.key_file = self.config_dir / "key.key"
        
        self.settings = SystemSettings()
        self._encryption_key = None
        
        if CRYPTO_AVAILABLE:
            self._ensure_encryption_key()
        self.load_settings()
    
    def _ensure_encryption_key(self):
        """Ensure encryption key exists."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available - settings will not be encrypted")
            return
            
        try:
            if self.key_file.exists():
                with open(self.key_file, 'rb') as f:
                    self._encryption_key = f.read()
            else:
                self._encryption_key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(self._encryption_key)
                # Set restrictive permissions
                os.chmod(self.key_file, 0o600)
        except Exception as e:
            logger.error(f"Failed to setup encryption key: {e}")
            if CRYPTO_AVAILABLE:
                self._encryption_key = Fernet.generate_key()
    
    def _encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        if not CRYPTO_AVAILABLE or not self._encryption_key:
            return data.encode()
            
        try:
            fernet = Fernet(self._encryption_key)
            return fernet.encrypt(data.encode())
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data.encode()
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        if not CRYPTO_AVAILABLE or not self._encryption_key:
            return encrypted_data.decode()
            
        try:
            fernet = Fernet(self._encryption_key)
            return fernet.decrypt(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""
    
    def load_settings(self) -> bool:
        """Load settings from configuration files."""
        try:
            # Load non-sensitive settings
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    self._apply_config_data(config_data)
            
            # Load encrypted sensitive settings
            self._load_encrypted_settings()
            
            logger.info("Settings loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to settings."""
        try:
            # Performance settings
            if "performance" in config_data:
                perf_data = config_data["performance"]
                for key, value in perf_data.items():
                    if hasattr(self.settings.performance, key):
                        setattr(self.settings.performance, key, value)
            
            # Risk settings
            if "risk" in config_data:
                risk_data = config_data["risk"]
                for key, value in risk_data.items():
                    if hasattr(self.settings.risk, key):
                        setattr(self.settings.risk, key, value)
            
            # Trading settings
            if "trading" in config_data:
                trading_data = config_data["trading"]
                for key, value in trading_data.items():
                    if hasattr(self.settings.trading, key):
                        setattr(self.settings.trading, key, value)
                        
        except Exception as e:
            logger.error(f"Failed to apply config data: {e}")
    
    def _load_encrypted_settings(self):
        """Load encrypted sensitive settings."""
        try:
            if self.encrypted_file.exists():
                with open(self.encrypted_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self._decrypt_data(encrypted_data)
                if decrypted_data:
                    api_data = json.loads(decrypted_data)
                    
                    for key, value in api_data.items():
                        if hasattr(self.settings.api, key):
                            setattr(self.settings.api, key, value)
                            
        except Exception as e:
            logger.error(f"Failed to load encrypted settings: {e}")
    
    def save_settings(self) -> bool:
        """Save settings to configuration files."""
        try:
            # Save non-sensitive settings to YAML
            config_data = {
                "performance": asdict(self.settings.performance),
                "risk": asdict(self.settings.risk),
                "trading": asdict(self.settings.trading)
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            # Save sensitive API settings encrypted
            self._save_encrypted_settings()
            
            logger.info("Settings saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def _save_encrypted_settings(self):
        """Save encrypted sensitive settings."""
        try:
            api_data = asdict(self.settings.api)
            api_json = json.dumps(api_data)
            encrypted_data = self._encrypt_data(api_json)
            
            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.encrypted_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save encrypted settings: {e}")
    
    def update_api_settings(self, **kwargs):
        """Update API settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.api, key):
                setattr(self.settings.api, key, value)
                logger.info(f"Updated API setting: {key}")
    
    def update_performance_settings(self, **kwargs):
        """Update performance settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.performance, key):
                setattr(self.settings.performance, key, value)
                logger.info(f"Updated performance setting: {key}")
    
    def update_risk_settings(self, **kwargs):
        """Update risk management settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.risk, key):
                setattr(self.settings.risk, key, value)
                logger.info(f"Updated risk setting: {key}")
    
    def update_trading_settings(self, **kwargs):
        """Update trading settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.trading, key):
                setattr(self.settings.trading, key, value)
                logger.info(f"Updated trading setting: {key}")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = SystemSettings()
        logger.info("Settings reset to defaults")
    
    def validate_settings(self) -> Dict[str, List[str]]:
        """Validate current settings and return any errors."""
        errors: Dict[str, List[str]] = {
            "api": [],
            "performance": [],
            "risk": [],
            "trading": []
        }
        
        # Validate API settings
        if self.settings.trading.trading_mode == "live":
            if not self.settings.api.coinbase_api_key:
                errors["api"].append("API key is required for live trading")
            if not self.settings.api.coinbase_secret:
                errors["api"].append("API secret is required for live trading")
        
        # Validate performance settings
        if self.settings.performance.cpu_threads < 1:
            errors["performance"].append("CPU threads must be at least 1")
        if self.settings.performance.memory_limit_mb < 512:
            errors["performance"].append("Memory limit must be at least 512 MB")
        
        # Validate risk settings
        if self.settings.risk.max_position_size > 1.0:
            errors["risk"].append("Max position size cannot exceed 100%")
        if self.settings.risk.stop_loss_percent > 0.5:
            errors["risk"].append("Stop loss percentage too high")
        
        # Validate trading settings
        if self.settings.trading.min_trade_amount <= 0:
            errors["trading"].append("Minimum trade amount must be positive")
        if self.settings.trading.max_concurrent_trades < 1:
            errors["trading"].append("Must allow at least 1 concurrent trade")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "api": {
                "sandbox_mode": self.settings.api.sandbox_mode,
                "has_credentials": bool(self.settings.api.coinbase_api_key),
                "timeout": self.settings.api.api_timeout
            },
            "performance": asdict(self.settings.performance),
            "risk": asdict(self.settings.risk),
            "trading": asdict(self.settings.trading)
        }
    
    def export_config(self, filepath: str, include_sensitive: bool = False) -> bool:
        """Export configuration to file."""
        try:
            config_data = {
                "performance": asdict(self.settings.performance),
                "risk": asdict(self.settings.risk),
                "trading": asdict(self.settings.trading)
            }
            
            if include_sensitive:
                config_data["api"] = asdict(self.settings.api)
            
            with open(filepath, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, filepath: str) -> bool:
        """Import configuration from file."""
        try:
            with open(filepath, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self._apply_config_data(config_data)
            
            logger.info(f"Configuration imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager(config_dir: Optional[str] = None) -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager(config_dir)
    return _settings_manager


def get_settings() -> SystemSettings:
    """Get the current system settings."""
    return get_settings_manager().settings


if __name__ == "__main__":
    # Test the settings manager
    manager = SettingsManager()
    
    # Update some settings
    manager.update_api_settings(
        coinbase_api_key="test_key",
        sandbox_mode=True
    )
    
    manager.update_trading_settings(
        trading_mode="demo",
        max_concurrent_trades=3
    )
    
    # Save settings
    manager.save_settings()
    
    # Validate settings
    errors = manager.validate_settings()
    print(f"Validation errors: {errors}")
    
    # Get config summary
    summary = manager.get_config_summary()
    print(f"Config summary: {summary}")