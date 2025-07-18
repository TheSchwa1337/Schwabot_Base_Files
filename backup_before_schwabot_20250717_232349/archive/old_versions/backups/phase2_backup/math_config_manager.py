#!/usr/bin/env python3
"""
Centralized Math Configuration Manager
Manages all mathematical operations configuration for Schwabot.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class MathConfigManager:
    """Centralized configuration manager for all mathematical operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/math_config.json"
        self.config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default mathematical configuration."""
        return {
            "precision": "float64",
            "cache_enabled": True,
            "gpu_acceleration": True,
            "parallel_processing": True,
            "optimization_level": "high",
            "max_cache_size": 10000,
            "cache_ttl": 3600,  # seconds
            "numerical_stability": {
                "epsilon": 1e-15,
                "max_iterations": 1000,
                "convergence_threshold": 1e-10
            },
            "tensor_operations": {
                "default_dtype": "float64",
                "memory_efficient": True,
                "chunk_size": 1000
            },
            "quantum_operations": {
                "simulation_precision": "double",
                "max_qubits": 32,
                "noise_model": "depolarizing"
            },
            "entropy_operations": {
                "base": 2,  # log base for entropy calculations
                "smoothing_factor": 1e-10,
                "normalization": True
            },
            "optimization": {
                "algorithm": "L-BFGS-B",
                "max_iterations": 1000,
                "tolerance": 1e-8,
                "constraints": "box"
            }
        }
    
    def _load_config(self):
        """Load configuration from file if it exists."""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                print(f"Loaded math config from: {self.config_path}")
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
    
    def save_config(self):
        """Save current configuration to file."""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Saved math config to: {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_precision(self) -> str:
        """Get numerical precision setting."""
        return self.get("precision", "float64")
    
    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.get("cache_enabled", True)
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self.get("gpu_acceleration", True)
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "enabled": self.is_cache_enabled(),
            "max_size": self.get("max_cache_size", 10000),
            "ttl": self.get("cache_ttl", 3600)
        }
    
    def get_numerical_stability_config(self) -> Dict[str, Any]:
        """Get numerical stability configuration."""
        return self.get("numerical_stability", {})

# Global instance
math_config = MathConfigManager()

def get_math_config() -> MathConfigManager:
    """Get the global math configuration manager."""
    return math_config
