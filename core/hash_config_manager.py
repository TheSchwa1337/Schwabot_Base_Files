#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hash Configuration Manager for Schwabot AI
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HashConfigManager:
    """Hash configuration manager for Schwabot AI."""
    
    def __init__(self, config_path: str = "config/hash_config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = self.get_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self.get_default_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "hash_algorithm": "sha256",
            "salt_length": 32,
            "iterations": 100000,
            "key_length": 64
        }
    
    def get_config(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
        self.save_config()
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required_keys = ["hash_algorithm", "salt_length", "iterations", "key_length"]
        return all(key in self.config for key in required_keys)

# Test function
def test_hash_config_manager():
    """Test the hash config manager."""
    try:
        manager = HashConfigManager()
        if manager.validate_config():
            print("Hash Config Manager: OK")
            return True
        else:
            print("Hash Config Manager: Configuration validation failed")
            return False
    except Exception as e:
        print(f"Hash Config Manager: Error - {e}")
        return False

if __name__ == "__main__":
    test_hash_config_manager()
