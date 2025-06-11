"""
Configuration Unifier Module
=========================

Centralizes all configuration management for Schwabot.
Provides unified interface for config loading, validation, and monitoring.
Supports runtime injection, auto-regeneration, and config state tracking.

Future Integrations:
------------------
1. Webhook Support:
   - Real-time config updates via webhooks
   - Config validation endpoints
   - Change notification system

2. Cluster Support:
   - Multi-node config synchronization
   - Config version control across cluster
   - Conflict resolution strategies

3. UI Dashboard:
   - Real-time config visualization
   - Config modification interface
   - Change history and rollback

4. AI Integration:
   - Config optimization suggestions
   - Performance correlation analysis
   - Automated config tuning

5. Security Features:
   - Config encryption at rest
   - Access control and audit logging
   - Config integrity verification
"""

import yaml
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ConfigState:
    """Tracks state of a configuration"""
    hash: str
    last_modified: datetime
    is_valid: bool = True
    validation_errors: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)

class ConfigUnifier:
    """Unified configuration management system"""
    
    def __init__(self, config_root: Optional[Path] = None):
        """Initialize config unifier"""
        self.root = config_root or Path(__file__).resolve().parent.parent / "config"
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Initialize config tracking
        self.configs: Dict[str, Dict] = {}
        self.config_states: Dict[str, ConfigState] = {}
        self._lock = threading.Lock()
        
        # Register managed configs
        self.managed_configs = {
            "matrix": self._register("matrix_response_paths.yaml", self._default_matrix),
            "line_render": self._register("line_render_config.yaml", self._default_line_render),
            "tesseract": self._register("tesseract_config.yaml", self._default_tesseract),
            "meta": self._register("schwabot_meta.yaml", self._default_meta),
            "phase": self._register("phase_config.json", self._default_phase),
            "strategy": self._register("strategy_config.yaml", self._default_strategy)
        }
        
        # Initialize config monitoring
        self._start_config_monitor()
        
    def _register(self, filename: str, default_func: Callable) -> Dict:
        """Register a config file with its default generator"""
        return {
            "path": self.root / filename,
            "default": default_func,
            "format": "yaml" if filename.endswith(".yaml") else "json"
        }
        
    def ensure_all(self) -> None:
        """Ensure all configs exist and are loaded"""
        with self._lock:
            for key, val in self.managed_configs.items():
                path = val["path"]
                if not path.exists():
                    logger.info(f"Creating default config: {key}")
                    val["default"](path)
                    
                # Load and validate config
                self._load_and_validate(key, path, val["format"])
                
    def _load_and_validate(self, key: str, path: Path, format: str) -> None:
        """Load and validate a config file"""
        try:
            if format == "yaml":
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open(path, 'r') as f:
                    config = json.load(f)
                    
            # Calculate config hash
            config_hash = self._hash_config(config)
            
            # Update config state
            self.config_states[key] = ConfigState(
                hash=config_hash,
                last_modified=datetime.fromtimestamp(path.stat().st_mtime)
            )
            
            # Store config
            self.configs[key] = config
            
            logger.info(f"Loaded config: {key} (hash: {config_hash[:8]})")
            
        except Exception as e:
            logger.error(f"Error loading config {key}: {e}")
            raise
            
    def get(self, key: str) -> Dict:
        """Get a config by key"""
        with self._lock:
            return self.configs.get(key, {})
            
    def get_state(self, key: str) -> Optional[ConfigState]:
        """Get state of a config"""
        return self.config_states.get(key)
        
    def _hash_config(self, config: Dict) -> str:
        """Calculate hash of config"""
        # Convert to canonical form
        if isinstance(config, dict):
            config_str = yaml.safe_dump(config, sort_keys=True)
        else:
            config_str = str(config)
            
        return hashlib.sha256(config_str.encode()).hexdigest()
        
    def _start_config_monitor(self) -> None:
        """Start background config monitoring"""
        def monitor():
            while True:
                try:
                    self._check_config_changes()
                    threading.Event().wait(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Config monitor error: {e}")
                    
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        
    def _check_config_changes(self) -> None:
        """Check for config file changes"""
        with self._lock:
            for key, val in self.managed_configs.items():
                path = val["path"]
                if not path.exists():
                    continue
                    
                # Check if file was modified
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                state = self.config_states.get(key)
                
                if state and mtime > state.last_modified:
                    logger.info(f"Config changed: {key}")
                    self._load_and_validate(key, path, val["format"])
                    
    def _default_matrix(self, path: Path) -> None:
        """Create default matrix response config"""
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
        self._write_config(path, config)
        
    def _default_line_render(self, path: Path) -> None:
        """Create default line render config"""
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
        self._write_config(path, config)
        
    def _default_tesseract(self, path: Path) -> None:
        """Create default tesseract config"""
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
        self._write_config(path, config)
        
    def _default_meta(self, path: Path) -> None:
        """Create default meta config"""
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
        self._write_config(path, config)
        
    def _default_phase(self, path: Path) -> None:
        """Create default phase config"""
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
        self._write_config(path, config)
        
    def _default_strategy(self, path: Path) -> None:
        """Create default strategy config"""
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
        self._write_config(path, config)
        
    def _write_config(self, path: Path, config: Dict) -> None:
        """Write config to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.yaml':
            with open(path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
                
        logger.info(f"Created default config: {path}") 