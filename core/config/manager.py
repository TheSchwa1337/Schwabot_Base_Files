"""
Central Config Manager
===================

Unifies configuration management across Schwabot.
Handles loading, validation, and dynamic updates of all configs.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import threading

logger = logging.getLogger(__name__)

class ConfigManager:
    """Central configuration management system"""
    
    def __init__(self, config_root: Optional[Path] = None):
        """Initialize the config manager"""
        self.root = config_root or Path(__file__).parent
        self.configs: Dict[str, Dict] = {}
        self.config_hashes: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._load_all_configs()
        
    def _load_all_configs(self) -> None:
        """Load all configuration files"""
        config_files = [
            "schema.yaml",
            "matrix_response_paths.yaml",
            "line_render_config.yaml",
            "phase_config.json",
            "strategy_config.yaml",
            "tesseract_config.yaml",
            "meta_config.yaml"
        ]
        
        for config_file in config_files:
            self._load_or_create_config(config_file)
            
    def _load_or_create_config(self, config_name: str) -> None:
        """Load or create a configuration file"""
        config_path = self.root / config_name
        
        try:
            if not config_path.exists():
                logger.info(f"Creating default config: {config_name}")
                self._create_default_config(config_name)
                
            # Load config
            if config_name.endswith('.yaml'):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
            # Calculate and store hash
            config_hash = self._hash_config(config)
            self.config_hashes[config_name] = config_hash
            
            # Store config
            self.configs[config_name] = config
            
            logger.info(f"Loaded config: {config_name} (hash: {config_hash[:8]})")
            
        except Exception as e:
            logger.error(f"Error loading config {config_name}: {e}")
            raise
            
    def _create_default_config(self, config_name: str) -> None:
        """Create a default configuration file"""
        config_path = self.root / config_name
        
        if config_name == "schema.yaml":
            self._create_schema_config(config_path)
        elif config_name == "matrix_response_paths.yaml":
            self._create_matrix_config(config_path)
        elif config_name == "line_render_config.yaml":
            self._create_line_render_config(config_path)
        elif config_name == "phase_config.json":
            self._create_phase_config(config_path)
        elif config_name == "strategy_config.yaml":
            self._create_strategy_config(config_path)
        elif config_name == "tesseract_config.yaml":
            self._create_tesseract_config(config_path)
        elif config_name == "meta_config.yaml":
            self._create_meta_config(config_path)
        else:
            raise ValueError(f"Unknown config type: {config_name}")
            
    def _hash_config(self, config: Dict) -> str:
        """Calculate hash of configuration"""
        config_str = yaml.safe_dump(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
        
    def get_config(self, config_name: str) -> Dict:
        """Get a configuration by name"""
        with self._lock:
            return self.configs.get(config_name, {}).copy()
            
    def get_config_hash(self, config_name: str) -> str:
        """Get hash of a configuration"""
        return self.config_hashes.get(config_name, "")
        
    def reload_config(self, config_name: str) -> None:
        """Reload a specific configuration"""
        with self._lock:
            self._load_or_create_config(config_name)
            
    def reload_all(self) -> None:
        """Reload all configurations"""
        with self._lock:
            self._load_all_configs()
            
    def get_path(self, config_name: str) -> Path:
        """Get path to a configuration file"""
        return self.root / config_name
        
    def _create_schema_config(self, path: Path) -> None:
        """Create default schema configuration"""
        config = {
            "phases": {
                "STABLE": {
                    "profit_trend_range": [0.001, float('inf')],
                    "stability_range": [0.7, 1.0],
                    "memory_coherence_range": [0.8, 1.0],
                    "paradox_pressure_range": [0.0, 2.0],
                    "entropy_rate_range": [0.0, 0.3],
                    "thermal_state_range": [0.0, 0.6],
                    "bit_depth_range": [16, 81],
                    "trust_score_range": [0.7, 1.0]
                }
            }
        }
        self._write_yaml_config(path, config)
        
    def _create_matrix_config(self, path: Path) -> None:
        """Create default matrix configuration"""
        config = {
            "matrix_response_paths": {
                "strategy_a": "strategies/alt_path_A.json",
                "fallback": "strategies/fallback_map.json"
            },
            "data_directory": "data/matrix_logs",
            "response_templates": {
                "safe": "hold",
                "warn": "delay_entry",
                "fail": "matrix_realign",
                "ZPE-risk": "cooldown_abort"
            }
        }
        self._write_yaml_config(path, config)
        
    def _create_line_render_config(self, path: Path) -> None:
        """Create default line render configuration"""
        config = {
            "style": "default",
            "dimensions": [1280, 720],
            "colors": {
                "stable": "#00ff00",
                "unstable": "#ff0000",
                "smart_money": "#0000ff",
                "overloaded": "#ff00ff"
            },
            "animation": {
                "enabled": True,
                "fps": 30,
                "transition_ms": 500
            }
        }
        self._write_yaml_config(path, config)
        
    def _create_phase_config(self, path: Path) -> None:
        """Create default phase configuration"""
        config = {
            "phase_regions": {
                "STABLE": {
                    "profit_trend_range": [0.001, float('inf')],
                    "stability_range": [0.7, 1.0],
                    "memory_coherence_range": [0.8, 1.0],
                    "paradox_pressure_range": [0.0, 2.0],
                    "entropy_rate_range": [0.0, 0.3],
                    "thermal_state_range": [0.0, 0.6],
                    "bit_depth_range": [16, 81],
                    "trust_score_range": [0.7, 1.0]
                }
            }
        }
        self._write_json_config(path, config)
        
    def _create_strategy_config(self, path: Path) -> None:
        """Create default strategy configuration"""
        config = {
            "active_strategies": ["default"],
            "default_strategy": {
                "type": "phase_aware",
                "parameters": {
                    "bit_depth": 64,
                    "trust_threshold": 0.7,
                    "phase_urgency_threshold": 0.5
                }
            },
            "baskets": {
                "BTC_USDC": {
                    "active": True,
                    "fallback_pairs": ["BTC_ETH", "ETH_USDC"]
                }
            }
        }
        self._write_yaml_config(path, config)
        
    def _create_tesseract_config(self, path: Path) -> None:
        """Create default tesseract configuration"""
        config = {
            "dimension": 8,
            "entropy_threshold": {
                "low": 0.15,
                "high": 0.92
            },
            "stability_mode": "vectorized",
            "use_gpu": True,
            "drift_detection": {
                "window_size": 1000,
                "threshold": 0.7
            }
        }
        self._write_yaml_config(path, config)
        
    def _create_meta_config(self, path: Path) -> None:
        """Create default meta configuration"""
        config = {
            "id": "schwabot_v043_sync",
            "risk_profile": "adaptive",
            "hash_mode": "SHA256_recursive",
            "logging": {
                "level": "DEBUG",
                "file": "schwabot.log",
                "rotation": {
                    "max_size": "100MB",
                    "backup_count": 5
                }
            },
            "sync": {
                "entropy_drift": True,
                "echo_memory": True,
                "interval_ms": 1000
            },
            "security": {
                "encryption_enabled": False,
                "access_control": {
                    "enabled": False,
                    "allowed_ips": []
                }
            }
        }
        self._write_yaml_config(path, config)
        
    def _write_yaml_config(self, path: Path, config: Dict) -> None:
        """Write YAML configuration to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
            
    def _write_json_config(self, path: Path, config: Dict) -> None:
        """Write JSON configuration to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
            
    def get_config_state(self) -> Dict[str, Any]:
        """Get current state of all configurations"""
        return {
            "configs": {
                name: {
                    "hash": self.config_hashes.get(name, ""),
                    "last_modified": datetime.fromtimestamp(
                        (self.root / name).stat().st_mtime
                    ).isoformat() if (self.root / name).exists() else None
                }
                for name in self.configs.keys()
            },
            "timestamp": datetime.now().isoformat()
        } 