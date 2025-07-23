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
    
    def initialize(self, cli_truncated_hash: bool = False, cli_hash_length: int = None):
        """Initialize the hash configuration manager with CLI options."""
        try:
            # Update config with CLI options if provided
            if cli_truncated_hash is not None:
                self.config['truncated_hash'] = cli_truncated_hash
            
            if cli_hash_length is not None:
                self.config['hash_length'] = cli_hash_length
            
            # Save updated config
            self.save_config()
            
            logger.info("Hash configuration manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize hash configuration manager: {e}")
            return False
    
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
            "key_length": 64,
            "truncated_hash": False,
            "hash_length": 32,
            # Kaprekar integration settings
            "kaprekar_enabled": True,
            "kaprekar_confidence_threshold": 0.7,
            "kaprekar_entropy_weight": 0.3,
            "kaprekar_strategy_boost": True,
            "kaprekar_max_steps": 7,
            "kaprekar_reject_threshold": 99
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

# Global instance for main.py import
hash_config_manager = HashConfigManager()

def get_hash_settings() -> Dict[str, Any]:
    """Get hash settings for use in other modules."""
    return hash_config_manager.config

def generate_hash_from_string(input_string: str) -> str:
    """
    Generate a hash from a string input.
    
    Args:
        input_string: String to hash
        
    Returns:
        Generated hash string
    """
    try:
        import hashlib
        
        # Get hash settings
        settings = get_hash_settings()
        algorithm = settings.get('hash_algorithm', 'sha256')
        
        # Generate hash
        if algorithm == 'sha256':
            hash_object = hashlib.sha256(input_string.encode('utf-8'))
        elif algorithm == 'sha512':
            hash_object = hashlib.sha512(input_string.encode('utf-8'))
        elif algorithm == 'md5':
            hash_object = hashlib.md5(input_string.encode('utf-8'))
        else:
            # Default to sha256
            hash_object = hashlib.sha256(input_string.encode('utf-8'))
        
        hash_hex = hash_object.hexdigest()
        
        # Truncate if configured
        if settings.get('truncated_hash', False):
            hash_length = settings.get('hash_length', 32)
            hash_hex = hash_hex[:hash_length]
        
        return hash_hex
        
    except Exception as e:
        logger.error(f"Error generating hash from string: {e}")
        # Return a fallback hash
        return "00000000000000000000000000000000"

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
