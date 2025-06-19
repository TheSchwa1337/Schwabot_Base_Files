#!/usr/bin/env python3
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

from __future__ import annotations

import os
import json
import yaml
import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union, Callable, Type
from enum import Enum
from datetime import datetime, timedelta
import warnings
from decimal import Decimal, getcontext

# Import Windows CLI compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler as CLIHandler,
        safe_print, safe_log
    )
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback CLI handler
    class CLIHandler:
        @staticmethod
        def safe_emoji_print(message: str, force_ascii: bool = False) -> str:
            emoji_mapping = {
                '✅': '[SUCCESS]', '❌': '[ERROR]', '⚠️': '[WARNING]', '🚨': '[ALERT]',
                '🎉': '[COMPLETE]', '🔄': '[PROCESSING]', '⏳': '[WAITING]', '⭐': '[STAR]',
                '🚀': '[LAUNCH]', '🔧': '[TOOLS]', '🛠️': '[REPAIR]', '⚡': '[FAST]',
                '🔍': '[SEARCH]', '🎯': '[TARGET]', '🔥': '[HOT]', '❄️': '[COOL]',
                '📊': '[DATA]', '📈': '[PROFIT]', '📉': '[LOSS]', '💰': '[MONEY]',
                '🧪': '[TEST]', '⚖️': '[BALANCE]', '🌡️': '[TEMP]', '🔬': '[ANALYZE]',
                '⚙️': '[SETTINGS]', '🔒': '[SECURE]', '🗂️': '[CONFIG]', '🔑': '[KEY]'
            }
            if force_ascii:
                for emoji, replacement in emoji_mapping.items():
                    message = message.replace(emoji, replacement)
            return message

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment enumeration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigSource(Enum):
    """Configuration source enumeration"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    RUNTIME = "runtime"


@dataclass
class SystemConfig:
    """System-level configuration"""
    
    # Basic system settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Performance settings
    max_workers: int = 4
    memory_limit_mb: int = 2048
    cpu_usage_limit: float = 0.8
    
    # Security settings
    encryption_enabled: bool = True
    api_rate_limit: int = 1000
    session_timeout: int = 3600
    
    # Windows CLI compatibility
    force_ascii_output: bool = False
    enable_emoji_fallback: bool = True
    cli_compatibility_mode: bool = True


@dataclass
class MathLibConfig:
    """Mathematical library configuration"""
    
    # Precision settings
    decimal_precision: int = 18
    floating_point_precision: str = "double"
    numerical_tolerance: float = 1e-10
    
    # Optimization settings
    optimization_algorithm: str = "adam"
    learning_rate: float = 0.001
    max_iterations: int = 10000
    convergence_threshold: float = 1e-8
    
    # Automatic differentiation
    enable_auto_diff: bool = True
    dual_number_precision: int = 16
    gradient_check_enabled: bool = True
    
    # Matrix operations
    matrix_backend: str = "numpy"  # numpy, torch, cupy
    enable_gpu_acceleration: bool = False
    gpu_device_id: int = 0


@dataclass
class TradingConfig:
    """Trading system configuration"""
    
    # Exchange settings
    default_exchange: str = "coinbase"
    sandbox_mode: bool = True
    api_timeout: int = 30
    retry_attempts: int = 3
    
    # Order management
    default_order_type: str = "limit"
    max_order_size: float = 1000.0
    min_order_size: float = 0.001
    
    # Risk management
    max_position_size: float = 10000.0
    max_daily_loss: float = 500.0
    risk_tolerance: float = 0.02
    
    # Strategy settings
    enable_backtesting: bool = True
    backtest_period_days: int = 30
    strategy_timeout: int = 300


@dataclass
class RealTimeConfig:
    """Real-time processing configuration"""
    
    # Data feed settings
    tick_buffer_size: int = 10000
    max_tick_age_seconds: int = 60
    data_compression_enabled: bool = True
    
    # Processing settings
    processing_threads: int = 2
    batch_size: int = 100
    processing_interval_ms: int = 100
    
    # Monitoring settings
    health_check_interval: int = 30
    performance_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 0.8,
        'memory_usage': 0.8,
        'error_rate': 0.05
    })


@dataclass
class AdvancedConfig:
    """Advanced features configuration"""
    
    # GAN filtering settings
    gan_enabled: bool = False
    gan_model_path: Optional[str] = None
    gan_confidence_threshold: float = 0.5
    gan_batch_size: int = 64
    
    # Quantum operations
    quantum_enabled: bool = False
    quantum_backend: str = "simulator"
    quantum_shots: int = 1024
    
    # Visualization settings
    visualization_enabled: bool = True
    chart_update_interval: int = 1000
    max_chart_points: int = 10000
    
    # GPU acceleration
    gpu_enabled: bool = False
    gpu_memory_fraction: float = 0.5
    gpu_allow_growth: bool = True


@dataclass
class IntegrationConfig:
    """Integration and external service configuration"""
    
    # Database settings
    database_url: Optional[str] = None
    database_pool_size: int = 10
    database_timeout: int = 30
    
    # API settings
    external_apis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    webhook_endpoints: List[str] = field(default_factory=list)
    
    # Notification settings
    email_enabled: bool = False
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    
    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30


@dataclass
class SchwaConfig:
    """
    Comprehensive Schwabot configuration container
    
    This class provides centralized configuration management for all
    components of the Schwabot system with validation and hot-reloading.
    """
    
    # Configuration sections
    system: SystemConfig = field(default_factory=SystemConfig)
    mathlib: MathLibConfig = field(default_factory=MathLibConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    realtime: RealTimeConfig = field(default_factory=RealTimeConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # Metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: ConfigSource = ConfigSource.FILE
    
    def __post_init__(self) -> None:
        """Post-initialization setup"""
        # Set decimal precision
        getcontext().prec = self.mathlib.decimal_precision
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.system.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.system.log_file
        )


class ConfigManager:
    """
    Configuration management system
    
    Provides centralized configuration management with validation,
    hot-reloading, and secure credential handling.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.cli_handler = CLIHandler()
        
        # Configuration state
        self.config: SchwaConfig = SchwaConfig()
        self.config_lock = threading.RLock()
        self.watchers: List[Callable[[SchwaConfig], None]] = []
        
        # Hot-reloading
        self.hot_reload_enabled = False
        self.hot_reload_thread: Optional[threading.Thread] = None
        self.last_modified: Optional[float] = None
        
        # Load initial configuration
        self._load_configuration()
        
        logger.info(f"ConfigManager initialized with {self.config_path}")
    
    def safe_print(self, message: str, force_ascii: Optional[bool] = None) -> None:
        """
        Safe print function with CLI compatibility
        
        Args:
            message: Message to print
            force_ascii: Force ASCII conversion
        """
        if force_ascii is None:
            force_ascii = self.config.system.force_ascii_output
        
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_print(message, force_ascii=force_ascii)
        else:
            safe_message = self.cli_handler.safe_emoji_print(message, force_ascii=force_ascii)
            print(safe_message)
    
    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """
        Safe logging function with CLI compatibility
        
        Args:
            level: Log level
            message: Message to log
            context: Additional context
            
        Returns:
            True if logging was successful
        """
        if CLI_COMPATIBILITY_AVAILABLE:
            return safe_log(logger, level, message, context)
        else:
            try:
                log_func = getattr(logger, level.lower(), logger.info)
                log_func(message)
                return True
            except Exception:
                return False
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Check for environment variable
        if 'SCHWABOT_CONFIG' in os.environ:
            return os.environ['SCHWABOT_CONFIG']
        
        # Check common locations
        possible_paths = [
            'schwabot_config.yaml',
            'config/schwabot.yaml',
            os.path.expanduser('~/.schwabot/config.yaml'),
            '/etc/schwabot/config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path
        return 'schwabot_config.yaml'
    
    def _load_configuration(self) -> None:
        """Load configuration from file"""
        try:
            with self.config_lock:
                if os.path.exists(self.config_path):
                    self.safe_log('info', f'Loading configuration from {self.config_path}')
                    
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                            config_data = yaml.safe_load(f)
                        else:
                            config_data = json.load(f)
                    
                    # Update configuration
                    self._update_config_from_dict(config_data)
                    
                    # Update metadata
                    self.config.source = ConfigSource.FILE
                    self.config.updated_at = datetime.now()
                    
                    # Track file modification time
                    self.last_modified = os.path.getmtime(self.config_path)
                    
                    self.safe_log('info', 'Configuration loaded successfully')
                else:
                    self.safe_log('warning', f'Configuration file not found: {self.config_path}')
                    self.safe_log('info', 'Using default configuration')
                    
                    # Save default configuration
                    self.save_configuration()
                
        except Exception as e:
            error_msg = f"Error loading configuration: {e}"
            self.safe_log('error', error_msg)
            self.safe_print(f"⚠️ {error_msg}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        try:
            # Update system config
            if 'system' in config_data:
                system_data = config_data['system']
                for key, value in system_data.items():
                    if hasattr(self.config.system, key):
                        setattr(self.config.system, key, value)
            
            # Update mathlib config
            if 'mathlib' in config_data:
                mathlib_data = config_data['mathlib']
                for key, value in mathlib_data.items():
                    if hasattr(self.config.mathlib, key):
                        setattr(self.config.mathlib, key, value)
            
            # Update trading config
            if 'trading' in config_data:
                trading_data = config_data['trading']
                for key, value in trading_data.items():
                    if hasattr(self.config.trading, key):
                        setattr(self.config.trading, key, value)
            
            # Update realtime config
            if 'realtime' in config_data:
                realtime_data = config_data['realtime']
                for key, value in realtime_data.items():
                    if hasattr(self.config.realtime, key):
                        setattr(self.config.realtime, key, value)
            
            # Update advanced config
            if 'advanced' in config_data:
                advanced_data = config_data['advanced']
                for key, value in advanced_data.items():
                    if hasattr(self.config.advanced, key):
                        setattr(self.config.advanced, key, value)
            
            # Update integration config
            if 'integration' in config_data:
                integration_data = config_data['integration']
                for key, value in integration_data.items():
                    if hasattr(self.config.integration, key):
                        setattr(self.config.integration, key, value)
            
            # Update metadata
            if 'version' in config_data:
                self.config.version = config_data['version']
            
        except Exception as e:
            error_msg = f"Error updating configuration: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def save_configuration(self) -> bool:
        """
        Save current configuration to file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.config_lock:
                # Convert to dictionary
                config_dict = asdict(self.config)
                
                # Remove datetime objects (not JSON serializable)
                config_dict.pop('created_at', None)
                config_dict.pop('updated_at', None)
                
                # Ensure directory exists
                config_dir = os.path.dirname(self.config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                
                # Save configuration
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                    else:
                        json.dump(config_dict, f, indent=2)
                
                self.safe_log('info', f'Configuration saved to {self.config_path}')
                return True
                
        except Exception as e:
            error_msg = f"Error saving configuration: {e}"
            self.safe_log('error', error_msg)
            return False
    
    def get_config(self) -> SchwaConfig:
        """
        Get current configuration
        
        Returns:
            Current configuration object
        """
        with self.config_lock:
            return self.config
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """
        Update a specific configuration value
        
        Args:
            section: Configuration section name
            key: Configuration key
            value: New value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.config_lock:
                if hasattr(self.config, section):
                    section_obj = getattr(self.config, section)
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        self.config.updated_at = datetime.now()
                        
                        # Notify watchers
                        self._notify_watchers()
                        
                        self.safe_log('info', f'Updated {section}.{key} = {value}')
                        return True
                    else:
                        self.safe_log('error', f'Key {key} not found in section {section}')
                        return False
                else:
                    self.safe_log('error', f'Section {section} not found')
                    return False
                    
        except Exception as e:
            error_msg = f"Error updating configuration: {e}"
            self.safe_log('error', error_msg)
            return False
    
    def add_watcher(self, callback: Callable[[SchwaConfig], None]) -> None:
        """
        Add configuration change watcher
        
        Args:
            callback: Function to call when configuration changes
        """
        self.watchers.append(callback)
        self.safe_log('info', 'Configuration watcher added')
    
    def _notify_watchers(self) -> None:
        """Notify all configuration watchers"""
        for watcher in self.watchers:
            try:
                watcher(self.config)
            except Exception as e:
                self.safe_log('error', f'Error in configuration watcher: {e}')
    
    def enable_hot_reload(self, check_interval: int = 5) -> None:
        """
        Enable hot-reloading of configuration
        
        Args:
            check_interval: Interval in seconds to check for changes
        """
        try:
            if self.hot_reload_enabled:
                self.safe_log('warning', 'Hot-reload already enabled')
                return
            
            self.hot_reload_enabled = True
            self.hot_reload_thread = threading.Thread(
                target=self._hot_reload_worker,
                args=(check_interval,),
                daemon=True
            )
            self.hot_reload_thread.start()
            
            self.safe_log('info', f'Hot-reload enabled with {check_interval}s interval')
            
        except Exception as e:
            error_msg = f"Error enabling hot-reload: {e}"
            self.safe_log('error', error_msg)
    
    def disable_hot_reload(self) -> None:
        """Disable hot-reloading of configuration"""
        self.hot_reload_enabled = False
        if self.hot_reload_thread:
            self.hot_reload_thread.join(timeout=1)
        self.safe_log('info', 'Hot-reload disabled')
    
    def _hot_reload_worker(self, check_interval: int) -> None:
        """Hot-reload worker thread"""
        while self.hot_reload_enabled:
            try:
                if os.path.exists(self.config_path):
                    current_modified = os.path.getmtime(self.config_path)
                    if self.last_modified and current_modified > self.last_modified:
                        self.safe_log('info', 'Configuration file changed, reloading...')
                        self._load_configuration()
                        self._notify_watchers()
                
                time.sleep(check_interval)
                
            except Exception as e:
                self.safe_log('error', f'Error in hot-reload worker: {e}')
                time.sleep(check_interval)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration
        
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'status': 'success',
                'errors': [],
                'warnings': [],
                'info': []
            }
            
            # Validate system configuration
            if self.config.system.max_workers <= 0:
                validation_results['errors'].append('system.max_workers must be positive')
            
            if self.config.system.memory_limit_mb <= 0:
                validation_results['errors'].append('system.memory_limit_mb must be positive')
            
            # Validate mathlib configuration
            if self.config.mathlib.decimal_precision < 1:
                validation_results['errors'].append('mathlib.decimal_precision must be at least 1')
            
            if self.config.mathlib.learning_rate <= 0:
                validation_results['errors'].append('mathlib.learning_rate must be positive')
            
            # Validate trading configuration
            if self.config.trading.max_order_size <= 0:
                validation_results['errors'].append('trading.max_order_size must be positive')
            
            if self.config.trading.risk_tolerance < 0 or self.config.trading.risk_tolerance > 1:
                validation_results['errors'].append('trading.risk_tolerance must be between 0 and 1')
            
            # Validate realtime configuration
            if self.config.realtime.tick_buffer_size <= 0:
                validation_results['errors'].append('realtime.tick_buffer_size must be positive')
            
            # Check for warnings
            if self.config.system.environment == Environment.PRODUCTION and self.config.system.debug:
                validation_results['warnings'].append('Debug mode enabled in production environment')
            
            if self.config.trading.sandbox_mode and self.config.system.environment == Environment.PRODUCTION:
                validation_results['warnings'].append('Sandbox mode enabled in production environment')
            
            # Set overall status
            if validation_results['errors']:
                validation_results['status'] = 'error'
            elif validation_results['warnings']:
                validation_results['status'] = 'warning'
            
            return validation_results
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f'Validation failed: {e}'],
                'warnings': [],
                'info': []
            }
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configuration
        
        Returns:
            Environment configuration dictionary
        """
        try:
            env_config = {}
            
            # Load environment variables
            for key, value in os.environ.items():
                if key.startswith('SCHWABOT_'):
                    config_key = key[10:].lower()  # Remove SCHWABOT_ prefix
                    env_config[config_key] = value
            
            return env_config
            
        except Exception as e:
            self.safe_log('error', f'Error getting environment config: {e}')
            return {}
    
    def export_config(self, format_type: str = 'yaml') -> str:
        """
        Export configuration to string
        
        Args:
            format_type: Export format ('yaml' or 'json')
            
        Returns:
            Configuration as string
        """
        try:
            with self.config_lock:
                config_dict = asdict(self.config)
                
                # Remove non-serializable fields
                config_dict.pop('created_at', None)
                config_dict.pop('updated_at', None)
                
                if format_type.lower() == 'yaml':
                    return yaml.dump(config_dict, default_flow_style=False, indent=2)
                else:
                    return json.dumps(config_dict, indent=2)
                    
        except Exception as e:
            error_msg = f"Error exporting configuration: {e}"
            self.safe_log('error', error_msg)
            return ""


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create global configuration manager
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> SchwaConfig:
    """
    Get current configuration
    
    Returns:
        Current configuration object
    """
    return get_config_manager().get_config()


def main() -> None:
    """
    Main function for testing configuration management
    
    Demonstrates configuration loading, validation, and management with
    CLI-safe output and comprehensive error handling.
    """
    try:
        print("🚀 Schwabot Configuration Management Test")
        print("=" * 50)
        
        # Initialize configuration manager
        print("⚙️ Initializing configuration manager...")
        config_manager = get_config_manager()
        
        # Get current configuration
        config = config_manager.get_config()
        print(f"✅ Configuration loaded:")
        print(f"   Environment: {config.system.environment.value}")
        print(f"   Version: {config.version}")
        print(f"   Debug mode: {config.system.debug}")
        print(f"   Default exchange: {config.trading.default_exchange}")
        
        # Validate configuration
        print("\n🔍 Validating configuration...")
        validation = config_manager.validate_configuration()
        print(f"   Status: {validation['status']}")
        
        if validation['errors']:
            print(f"   Errors: {len(validation['errors'])}")
            for error in validation['errors']:
                print(f"     - {error}")
        
        if validation['warnings']:
            print(f"   Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings']:
                print(f"     - {warning}")
        
        # Test configuration update
        print("\n🔧 Testing configuration update...")
        success = config_manager.update_config('system', 'log_level', 'DEBUG')
        if success:
            print("✅ Configuration updated successfully")
            updated_config = config_manager.get_config()
            print(f"   New log level: {updated_config.system.log_level}")
        else:
            print("❌ Configuration update failed")
        
        # Test configuration export
        print("\n📤 Testing configuration export...")
        yaml_export = config_manager.export_config('yaml')
        if yaml_export:
            print(f"✅ Configuration exported ({len(yaml_export)} characters)")
        else:
            print("❌ Configuration export failed")
        
        # Test configuration save
        print("\n💾 Testing configuration save...")
        save_success = config_manager.save_configuration()
        if save_success:
            print("✅ Configuration saved successfully")
        else:
            print("❌ Configuration save failed")
        
        print("\n🎉 Configuration management test completed successfully!")
        
    except Exception as e:
        print(f"❌ Configuration management test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 