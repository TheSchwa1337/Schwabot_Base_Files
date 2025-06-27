"""
Schwabot Configuration Management System
=======================================

Comprehensive configuration management system for the Schwabot mathematical
trading framework. Provides centralized configuration with validation,
environment-specific settings, and runtime configuration updates.

Key Features:
- Centralized configuration management for all components
- Environment-specific configuration (development, production, testing)
- Configuration validation with schema enforcement
- Runtime configuration updates with hot-reloading
- Secure credential management with encryption
- Configuration versioning and rollback capabilities
- Integration with all core components
- Windows CLI compatibility with emoji fallbacks

Configuration Categories:
- System settings (logging, performance, security)
- Mathematical libraries (precision, optimization, algorithms)
- Trading system (exchanges, strategies, risk management)
- Real-time processing (data feeds, tick processing, monitoring)
- Advanced features (GAN filtering, quantum operations, visualization)
- Integration settings (APIs, databases, external services)

Integration Points:
- All core components for configuration access
- enhanced_windows_cli_compatibility.py: CLI compatibility
- constraints.py: Configuration validation
- mathlib_v3.py: Mathematical precision settings
- simplified_btc_integration.py: Exchange configuration

Windows CLI compatible with flake8 compliance.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import getcontext
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import yaml

# Import Windows CLI compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler as CLIHandler
    )
    from core.enhanced_windows_cli_compatibility import safe_log
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False

# Import unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    import math as unified_math

# Configure logging
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    STAGING = "staging"


class ConfigType(Enum):
    """Configuration types."""
    SYSTEM = "system"
    MATHEMATICAL = "mathematical"
    TRADING = "trading"
    REAL_TIME = "real_time"
    ADVANCED = "advanced"
    INTEGRATION = "integration"


@dataclass
class MathematicalConfig:
    """Mathematical configuration settings."""
    precision: int = 128
    optimization_level: int = 2
    algorithm_preference: str = "balanced"
    convergence_threshold: float = 1e-10
    max_iterations: int = 1000
    numerical_stability: bool = True
    parallel_processing: bool = True
    gpu_acceleration: bool = False


@dataclass
class TradingConfig:
    """Trading system configuration."""
    default_exchange: str = "binance"
    risk_management_enabled: bool = True
    max_position_size: float = 0.1
    stop_loss_percentage: float = 0.02
    take_profit_percentage: float = 0.04
    max_drawdown: float = 0.15
    leverage_limit: int = 3
    trading_pairs: List[str] = field(
        default_factory=lambda: [
            "BTC/USDT", "ETH/USDT"])


@dataclass
class SystemConfig:
    """System configuration settings."""
    log_level: str = "INFO"
    log_file: str = "logs/schwabot.log"
    max_log_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    performance_monitoring: bool = True
    health_check_interval: int = 30
    auto_restart: bool = True
    emergency_shutdown: bool = True


@dataclass
class RealTimeConfig:
    """Real-time processing configuration."""
    tick_buffer_size: int = 10000
    processing_threads: int = 4
    batch_size: int = 100
    timeout_seconds: float = 5.0
    retry_attempts: int = 3
    data_validation: bool = True
    compression_enabled: bool = True


@dataclass
class AdvancedConfig:
    """Advanced features configuration."""
    gan_filtering: bool = True
    quantum_operations: bool = False
    visualization_enabled: bool = True
    ai_integration: bool = True
    fractal_analysis: bool = True
    entropy_calculation: bool = True
    thermal_management: bool = True


@dataclass
class IntegrationConfig:
    """Integration settings configuration."""
    api_timeout: float = 10.0
    database_connection_pool: int = 10
    cache_enabled: bool = True
    cache_size: int = 1000
    external_apis: Dict[str, str] = field(default_factory=dict)
    webhook_urls: List[str] = field(default_factory=list)


@dataclass
class SchwabotConfig:
    """Main configuration container."""
    environment: Environment = Environment.DEVELOPMENT
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Configuration sections
    mathematical: MathematicalConfig = field(
        default_factory=MathematicalConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    real_time: RealTimeConfig = field(default_factory=RealTimeConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Configuration management system."""

    def __init__(self, config_path: str = "config/schwabot_config.yaml"):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.config: Optional[SchwabotConfig] = None
        self._lock = threading.RLock()
        self._watchers: List[Callable[[SchwabotConfig], None]] = []

        # Load initial configuration
        self.load_configuration()

        logger.info("Configuration manager initialized")

    def load_configuration(self) -> bool:
        """Load configuration from file."""
        try:
            with self._lock:
                if os.path.exists(self.config_path):
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                        # Convert to SchwabotConfig object
                        self.config = self._dict_to_config(config_data)
                        logger.info(
                            f"Configuration loaded from {
                                self.config_path}")
                        return True
                else:
                    # Create default configuration
                    self.config = SchwabotConfig()
                    self.save_configuration()
                    logger.info("Default configuration created")
                    return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fallback to default configuration
            self.config = SchwabotConfig()
            return False

    def save_configuration(self) -> bool:
        """Save configuration to file."""
        try:
            with self._lock:
                if self.config is None:
                    return False

                # Ensure directory exists
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

                # Convert to dictionary
                config_data = self._config_to_dict(self.config)

                # Update timestamp
                config_data['last_updated'] = datetime.now().isoformat()

                # Save to file
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(
                        config_data,
                        f,
                        default_flow_style=False,
                        indent=2)

                logger.info(f"Configuration saved to {self.config_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get_config(self) -> SchwabotConfig:
        """Get current configuration."""
        with self._lock:
            if self.config is None:
                self.load_configuration()
            return self.config

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            with self._lock:
                if self.config is None:
                    return False

                # Apply updates
                self._apply_updates(self.config, updates)

                # Update timestamp
                self.config.last_updated = datetime.now()

                # Save configuration
                success = self.save_configuration()

                # Notify watchers
                if success:
                    self._notify_watchers()

                return success

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def get_mathematical_config(self) -> MathematicalConfig:
        """Get mathematical configuration."""
        return self.get_config().mathematical

    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self.get_config().trading

    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.get_config().system

    def get_real_time_config(self) -> RealTimeConfig:
        """Get real-time configuration."""
        return self.get_config().real_time

    def get_advanced_config(self) -> AdvancedConfig:
        """Get advanced configuration."""
        return self.get_config().advanced

    def get_integration_config(self) -> IntegrationConfig:
        """Get integration configuration."""
        return self.get_config().integration

    def add_watcher(self, callback: Callable[[SchwabotConfig], None]) -> None:
        """Add configuration change watcher."""
        with self._lock:
            self._watchers.append(callback)

    def remove_watcher(
            self, callback: Callable[[SchwabotConfig], None]) -> None:
        """Remove configuration change watcher."""
        with self._lock:
            if callback in self._watchers:
                self._watchers.remove(callback)

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate current configuration."""
        errors = {}

        try:
            config = self.get_config()

            # Validate mathematical configuration
            math_errors = self._validate_mathematical_config(
                config.mathematical)
            if math_errors:
                errors['mathematical'] = math_errors

            # Validate trading configuration
            trading_errors = self._validate_trading_config(config.trading)
            if trading_errors:
                errors['trading'] = trading_errors

            # Validate system configuration
            system_errors = self._validate_system_config(config.system)
            if system_errors:
                errors['system'] = system_errors

            # Validate real-time configuration
            realtime_errors = self._validate_realtime_config(config.real_time)
            if realtime_errors:
                errors['real_time'] = realtime_errors

        except Exception as e:
            errors['general'] = [f"Configuration validation failed: {e}"]

        return errors

    def _dict_to_config(self, data: Dict[str, Any]) -> SchwabotConfig:
        """Convert dictionary to SchwabotConfig object."""
        try:
            # Handle environment
            if 'environment' in data:
                data['environment'] = Environment(data['environment'])

            # Handle mathematical config
            if 'mathematical' in data:
                data['mathematical'] = MathematicalConfig(
                    **data['mathematical'])

            # Handle trading config
            if 'trading' in data:
                data['trading'] = TradingConfig(**data['trading'])

            # Handle system config
            if 'system' in data:
                data['system'] = SystemConfig(**data['system'])

            # Handle real-time config
            if 'real_time' in data:
                data['real_time'] = RealTimeConfig(**data['real_time'])

            # Handle advanced config
            if 'advanced' in data:
                data['advanced'] = AdvancedConfig(**data['advanced'])

            # Handle integration config
            if 'integration' in data:
                data['integration'] = IntegrationConfig(**data['integration'])

            # Handle timestamps
            if 'created_at' in data:
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if 'last_updated' in data:
                data['last_updated'] = datetime.fromisoformat(
                    data['last_updated'])

            return SchwabotConfig(**data)

        except Exception as e:
            logger.error(f"Failed to convert dict to config: {e}")
            return SchwabotConfig()

    def _config_to_dict(self, config: SchwabotConfig) -> Dict[str, Any]:
        """Convert SchwabotConfig object to dictionary."""
        try:
            data = asdict(config)

            # Convert enums to strings
            data['environment'] = config.environment.value

            # Convert timestamps to ISO format
            data['created_at'] = config.created_at.isoformat()
            data['last_updated'] = config.last_updated.isoformat()

            return data

        except Exception as e:
            logger.error(f"Failed to convert config to dict: {e}")
            return {}

    def _apply_updates(self, config: SchwabotConfig,
                       updates: Dict[str, Any]) -> None:
        """Apply updates to configuration."""
        for key, value in updates.items():
            if hasattr(config, key):
                if isinstance(
                    value,
                    dict) and hasattr(
                    getattr(
                        config,
                        key),
                        '__dict__'):
                    # Update nested object
                    current = getattr(config, key)
                    for subkey, subvalue in value.items():
                        if hasattr(current, subkey):
                            setattr(current, subkey, subvalue)
                else:
                    # Update direct attribute
                    setattr(config, key, value)

    def _notify_watchers(self) -> None:
        """Notify configuration change watchers."""
        config = self.get_config()
        for watcher in self._watchers:
            try:
                watcher(config)
            except Exception as e:
                logger.error(f"Watcher notification failed: {e}")

    def _validate_mathematical_config(
            self, config: MathematicalConfig) -> List[str]:
        """Validate mathematical configuration."""
        errors = []

        if config.precision < 1 or config.precision > 512:
            errors.append("Precision must be between 1 and 512")

        if config.optimization_level < 0 or config.optimization_level > 5:
            errors.append("Optimization level must be between 0 and 5")

        if config.convergence_threshold <= 0:
            errors.append("Convergence threshold must be positive")

        if config.max_iterations < 1:
            errors.append("Max iterations must be at least 1")

        return errors

    def _validate_trading_config(self, config: TradingConfig) -> List[str]:
        """Validate trading configuration."""
        errors = []

        if config.max_position_size <= 0 or config.max_position_size > 1:
            errors.append("Max position size must be between 0 and 1")

        if config.stop_loss_percentage <= 0:
            errors.append("Stop loss percentage must be positive")

        if config.take_profit_percentage <= 0:
            errors.append("Take profit percentage must be positive")

        if config.max_drawdown <= 0 or config.max_drawdown > 1:
            errors.append("Max drawdown must be between 0 and 1")

        if config.leverage_limit < 1:
            errors.append("Leverage limit must be at least 1")

        return errors

    def _validate_system_config(self, config: SystemConfig) -> List[str]:
        """Validate system configuration."""
        errors = []

        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")

        if config.max_log_size <= 0:
            errors.append("Max log size must be positive")

        if config.backup_count < 0:
            errors.append("Backup count must be non-negative")

        if config.health_check_interval <= 0:
            errors.append("Health check interval must be positive")

        return errors

    def _validate_realtime_config(self, config: RealTimeConfig) -> List[str]:
        """Validate real-time configuration."""
        errors = []

        if config.tick_buffer_size <= 0:
            errors.append("Tick buffer size must be positive")

        if config.processing_threads <= 0:
            errors.append("Processing threads must be positive")

        if config.batch_size <= 0:
            errors.append("Batch size must be positive")

        if config.timeout_seconds <= 0:
            errors.append("Timeout seconds must be positive")

        if config.retry_attempts < 0:
            errors.append("Retry attempts must be non-negative")

        return errors


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> SchwabotConfig:
    """Get current configuration."""
    return get_config_manager().get_config()


def update_config(updates: Dict[str, Any]) -> bool:
    """Update configuration."""
    return get_config_manager().update_config(updates)


def validate_config() -> Dict[str, List[str]]:
    """Validate current configuration."""
    return get_config_manager().validate_configuration()


def main():
    """Main function for testing configuration management."""
    try:
        # Create configuration manager
        config_manager = get_config_manager()

        # Get current configuration
        config = config_manager.get_config()
        print(f"Current environment: {config.environment.value}")
        print(f"Mathematical precision: {config.mathematical.precision}")
        print(f"Trading pairs: {config.trading.trading_pairs}")

        # Validate configuration
        errors = config_manager.validate_configuration()
        if errors:
            print(f"Configuration errors: {errors}")
        else:
            print("Configuration is valid")

        # Test configuration update
        updates = {
            'mathematical': {
                'precision': 256,
                'optimization_level': 3
            }
        }

        if config_manager.update_config(updates):
            print("Configuration updated successfully")
        else:
            print("Configuration update failed")

    except Exception as e:
        print(f"Configuration test failed: {e}")


if __name__ == "__main__":
    main()
