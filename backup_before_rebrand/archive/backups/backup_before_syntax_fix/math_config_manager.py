#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Config Manager Implementation ⚙️

Provides mathematical configuration management:
• Load and save mathematical configurations
• Manage hardware preferences (CPU/GPU)
• Cache configuration and performance settings
• Configuration validation and defaults

Features:
- Mathematical configuration management
- Hardware preference settings (CPU/GPU/Auto)
- Cache configuration and TTL management
- Configuration file persistence
- Validation and default settings
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MathConfigManager:
    """
    Mathematical configuration manager.
    Handles loading, saving, and managing mathematical system configurations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MathConfigManager with configuration."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        self.config_file = "config/math_config.json"
        
        # Initialize math infrastructure if available
        try:
            from core.math_orchestrator import MathOrchestrator
            self.math_orchestrator = MathOrchestrator()
            self.math_infrastructure_available = True
        except ImportError:
            self.math_infrastructure_available = False
            logger.warning("⚠️ Math orchestrator not available")
        
        self._initialize_system()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
            'hardware_preference': 'auto',  # 'cpu', 'gpu', 'auto'
            'cache_enabled': True,
            'cache_ttl': 3600,
            'max_cache_size': 1000,
            'parallel_processing': True,
            'batch_size': 100,
        }
    
    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            
            # Load config from file if it exists
            self._load_config()
            
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info(f"✅ {self.__class__.__name__} activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
            'math_infrastructure_available': self.math_infrastructure_available
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration."""
        if not self.active:
            self.logger.warning("Config manager not active")
            return {}
        
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if not self.active:
            self.logger.warning("Config manager not active")
            return default
        
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """Set a configuration value."""
        if not self.active:
            self.logger.warning("Config manager not active")
            return False
        
        self.config[key] = value
        self.logger.debug(f"Config set: {key} = {value}")
        return True
    
    def _load_config(self) -> bool:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                    self.logger.info(f"✅ Config loaded from {self.config_file}")
                    return True
            else:
                self.logger.info(f"Config file {self.config_file} not found, using defaults")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        if not self.active:
            self.logger.warning("Config manager not active")
            return False
        
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"✅ Config saved to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving config: {e}")
            return False
    
    def reload_config(self) -> bool:
        """Reload configuration from file."""
        if not self.active:
            self.logger.warning("Config manager not active")
            return False
        
        return self._load_config()
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        try:
            required_keys = ['enabled', 'timeout', 'retries', 'hardware_preference']
            
            for key in required_keys:
                if key not in self.config:
                    self.logger.error(f"❌ Missing required config key: {key}")
                    return False
            
            # Validate hardware preference
            valid_hardware = ['cpu', 'gpu', 'auto']
            if self.config['hardware_preference'] not in valid_hardware:
                self.logger.error(f"❌ Invalid hardware preference: {self.config['hardware_preference']}")
                return False
            
            # Validate numeric values
            if not isinstance(self.config['timeout'], (int, float)) or self.config['timeout'] <= 0:
                self.logger.error(f"❌ Invalid timeout value: {self.config['timeout']}")
                return False
            
            if not isinstance(self.config['retries'], int) or self.config['retries'] < 0:
                self.logger.error(f"❌ Invalid retries value: {self.config['retries']}")
                return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware-specific configuration."""
        try:
            hardware_pref = self.config.get('hardware_preference', 'auto')
            
            if hardware_pref == 'gpu':
                return {
                    'use_gpu': True,
                    'use_cpu': False,
                    'fallback_to_cpu': True
                }
            elif hardware_pref == 'cpu':
                return {
                    'use_gpu': False,
                    'use_cpu': True,
                    'fallback_to_cpu': False
                }
            else:  # auto
                return {
                    'use_gpu': True,
                    'use_cpu': True,
                    'fallback_to_cpu': True
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get hardware config: {e}")
            return {
                'use_gpu': False,
                'use_cpu': True,
                'fallback_to_cpu': False
            }


# Factory function
def create_math_config_manager(config: Optional[Dict[str, Any]] = None) -> MathConfigManager:
    """Create a math config manager instance."""
    return MathConfigManager(config)


# Singleton instance for global use
math_config_manager = MathConfigManager()
