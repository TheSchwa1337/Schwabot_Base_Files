"""
Tesseract Configuration Loader
Handles loading and merging of Tesseract configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from copy import deepcopy
import logging
import shutil
from datetime import datetime

# Setup logger
logger = logging.getLogger("TesseractConfig")

# Optional utility for deep dict merge
def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge dict2 into dict1."""
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            deep_merge(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1

class TesseractConfig:
    def __init__(self, config_dir: str = "core/config/tesseract"):
        self.config_dir = Path(config_dir)

        # Load all YAML config files
        self.quantum_config = self._load_config("quantum_config.yaml")
        self.pattern_config = self._load_config("pattern_config.yaml")
        self.risk_config = self._load_config("risk_config.yaml")
        self.monitoring_config = self._load_config("monitoring_config.yaml")
        self.propagation_config = self._load_config("propagation_config.yaml")

    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {filename}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {filename}: {e}")
            return {}

    def get_quantum_params(self) -> Dict[str, Any]:
        """Get quantum-cellular synchronization parameters."""
        return self.quantum_config.get('quantum', {})

    def get_pattern_params(self) -> Dict[str, Any]:
        """Get pattern analysis parameters."""
        return self.pattern_config.get('pattern_analysis', {})

    def get_risk_params(self) -> Dict[str, Any]:
        """Get risk management parameters."""
        return self.risk_config.get('risk_management', {})

    def get_monitoring_params(self) -> Dict[str, Any]:
        """Get system monitoring parameters."""
        return self.monitoring_config.get('monitoring', {})

    def get_propagation_params(self) -> Dict[str, Any]:
        """Get propagation engine parameters."""
        return self.propagation_config.get('core', {})

    def get_gan_params(self) -> Dict[str, Any]:
        """Get GAN configuration parameters."""
        return self.propagation_config.get('gan_config', {})

    def get_phase_params(self) -> Dict[str, Any]:
        """Get phase handling parameters."""
        return self.propagation_config.get('phase', {})

    def get_plasma_params(self) -> Dict[str, Any]:
        """Get plasma detection parameters."""
        return self.propagation_config.get('plasma', {})

    def get_archetype_params(self) -> Dict[str, Any]:
        """Get archetype matching parameters."""
        return self.propagation_config.get('archetype', {})

    def get_memory_params(self) -> Dict[str, Any]:
        """Get memory management parameters."""
        return self.propagation_config.get('memory', {})

    def get_integration_params(self) -> Dict[str, Any]:
        """Get integration parameters."""
        return self.propagation_config.get('integration', {})

    def get_safety_params(self) -> Dict[str, Any]:
        """Get safety threshold parameters."""
        return self.propagation_config.get('safety', {})

    def get_dimension_weights(self) -> Dict[str, float]:
        """Get the weights for each dimension in pattern analysis."""
        dimensions = self.pattern_config.get('pattern_analysis', {}).get('dimensions', {})
        return {dim: params.get('weight', 0.0) for dim, params in dimensions.items()}

    def get_risk_limits(self) -> Dict[str, Dict[str, float]]:
        """Get risk limits for different time periods."""
        return self.risk_config.get('risk_management', {}).get('risk_limits', {})

    def get_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get alert thresholds for different metrics."""
        return self.monitoring_config.get('monitoring', {}).get('alerts', {}).get('alert_levels', {})

    def get_visualization_params(self) -> Dict[str, Any]:
        """Get visualization parameters."""
        return self.pattern_config.get('pattern_analysis', {}).get('pattern_visualization', {})

    def get_logging_params(self) -> Dict[str, Any]:
        """Get logging parameters."""
        return self.monitoring_config.get('monitoring', {}).get('logging', {})

    def get_notification_channels(self) -> Dict[str, Dict[str, Any]]:
        """Get notification channel configurations."""
        return self.monitoring_config.get('monitoring', {}).get('alerts', {}).get('notification_channels', {})

    def validate_config(self) -> bool:
        """Validate the configuration for consistency and completeness."""
        required = {
            "quantum": self.get_quantum_params(),
            "pattern_analysis": self.get_pattern_params(),
            "risk_management": self.get_risk_params(),
            "monitoring": self.get_monitoring_params(),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"[TesseractConfig] Missing required config sections: {missing}")

        weights = self.get_dimension_weights()
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("[TesseractConfig] Dimension weights must sum to 1.0")

        logger.info("[TesseractConfig] Configuration validated successfully.")
        return True

    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a value inside the config (in-memory only)"""
        if section == "quantum":
            self.quantum_config.setdefault("quantum", {})[key] = value
        elif section == "pattern":
            self.pattern_config.setdefault("pattern_analysis", {})[key] = value
        elif section == "risk":
            self.risk_config.setdefault("risk_management", {})[key] = value
        elif section == "monitoring":
            self.monitoring_config.setdefault("monitoring", {})[key] = value
        else:
            raise ValueError(f"Invalid section: {section}")
        logger.info(f"[TesseractConfig] Updated {section}:{key} = {value}")

    def save_config(self) -> None:
        """Save all configuration changes to disk."""
        configs = {
            'quantum_config.yaml': self.quantum_config,
            'pattern_config.yaml': self.pattern_config,
            'risk_config.yaml': self.risk_config,
            'monitoring_config.yaml': self.monitoring_config
        }

        for filename, config in configs.items():
            config_path = self.config_dir / filename
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"[TesseractConfig] Saved config to {config_path}")

    @classmethod
    def create_default_config(cls, config_dir: str = "core/config/tesseract") -> 'TesseractConfig':
        """Create a new configuration with default values."""
        config = cls(config_dir)
        config.validate_config()
        return config

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"TesseractConfig(quantum={bool(self.quantum_config)}, pattern={bool(self.pattern_config)}, risk={bool(self.risk_config)}, monitoring={bool(self.monitoring_config)})"

    def get_basket_risk_config(self, basket_type: str) -> Dict[str, Any]:
        """Get risk configuration for specific basket type."""
        base_config = deepcopy(self.risk_config.get('risk_management', {}))
        basket_config = deepcopy(self.risk_config.get('risk_management', {}).get('risk_limits', {}).get(basket_type, {}))
        
        # Check if basket config exists
        if not basket_config:
            logger.warning(f"[TesseractConfig] No basket config found for: {basket_type}")
        
        return deep_merge(base_config, basket_config)

    def get_portfolio_risk_config(self) -> Dict[str, Any]:
        """Get portfolio-level risk configuration."""
        return deepcopy(self.risk_config.get('risk_management', {}).get('risk_limits', {}).get('aggregation', {}))

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory management configuration."""
        return deepcopy(self.risk_config.get('risk_management', {}).get('risk_limits', {}).get('monitoring', {})) 