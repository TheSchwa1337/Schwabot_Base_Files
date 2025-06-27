import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from dual_unicore_handler import DualUnicoreHandler
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import math
import os
import yaml

from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf - 8 -*-\\n# Import safe print for Windows compatibility
try:
# EMERGENCY:     """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""  # Original error: invalid syntax (<unknown>, line 21)
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
print("[INFO] {message}")


def warn(message):
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
print("[WARN] {message}")


def error(message):
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
print("[ERROR] {message}")


def success(message):
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
print("[SUCCESS] {message}")


def debug(message):
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
print("[DEBUG] {message}")


"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
entry_logic: str="tick + fractal"
exit_logic: str="hash + api_vol"
ghost_signal_weight: float=0.7
strict_mode_enabled: bool=True
tick_delta_threshold: float=0.2
volume_sync_enabled: bool=True
hash_confidence_min: float=0.6
api_echo_sync: bool=True


@dataclass
class Placeholder:
    pass  # Emergency placeholder

"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""
matrix_overlay: str="full"
entropy_trigger_threshold: float=0.2
learning_rate: float=0.5
memory_decay: float=0.95
success_reward: float=1.5
failure_penalty: float=0.92


@dataclass
class Placeholder:
    pass  # Emergency placeholder

"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""
"""
def __init__(self, config_path: str = "settings/"):
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.matrix_settings = MatrixSettings("SFS8 - A5")
        self.vector_settings = VectorSettings()
        self.allocator_settings = AllocatorSettings()
        self.allocator_settings.allocator_mode = ["long", "mid"]
self.reinforcement_settings = ReinforcementSettings()
        self.fault_settings = FaultSettings()

# Load settings from files
self._load_settings()

# Initialize known bad vectors map
self.known_bad_vectors = self._load_known_bad_vectors()

# Matrix path weights for reinforcement learning
self.matrix_path_weights = {}
self._initialize_matrix_weights()


def _load_settings(self) -> None:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""
main_config=self.config_path / "main_settings.yaml"
        if main_config.exists():
        with open(main_config, 'r') as f:
        config_data = yaml.safe_load(f)
        self._apply_main_settings(config_data)

# Load matrix - specific settings
matrix_config = self.config_path / "matrix_settings.yaml"
        if matrix_config.exists():
        with open(matrix_config, 'r') as f:
        matrix_data = yaml.safe_load(f)
        self._apply_matrix_settings(matrix_data)

# Load demo / test settings
_demo_config = self.config_path / "demo_backtest_mode.yaml"
        if demo_config.exists():
        with open(demo_config, 'r') as f:
        demo_data = yaml.safe_load(f)
        self._apply_demo_settings(demo_data)

except Exception as e:
    pass  # TODO: Implement except block
safe_print("Warning: Could not load settings: {e}")
        self._create_default_settings()

def _create_default_settings(self) -> None:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Create default settings files"""Emergency consolidated docstring."""Emergency consolidated docstring."""
main_settings={}"""
"mode": "production",
"matrix_mode": "42 - phase",
"enable_backlog_reinforcement": True,
"fault_tolerance": 0.15,
"api_echo_sync": True


with open(self.config_path / "main_settings.yaml", 'w') as f:
        yaml.dump(main_settings, f, default_flow_style = False)

# Demo settings
demo_settings = {}
"mode": "demo",
"backtest_path": "./tests / demo_backlog/",
"reinforce_bad_vectors": True,
"log_ghost_trades": True,
"matrix_overlay": "full",
"entropy_trigger_threshold": 0.2


with open(self.config_path / "demo_backtest_mode.yaml", 'w') as f:
        yaml.dump(demo_settings, f, default_flow_style = False)

def _apply_main_settings(self, config_data: Dict[str, Any]) -> None:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Apply main configuration settings"""Emergency consolidated docstring."""Emergency consolidated docstring."""
if "matrix_mode" in config_data:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.matrix_settings.matrix_id=config_data["matrix_mode"]

if "enable_backlog_reinforcement" in config_data:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.reinforcement_settings.enable_backlog_reinforcement=config_data["enable_backlog_reinforcement"]

if "fault_tolerance" in config_data:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.fault_settings.fault_tolerance=config_data["fault_tolerance"]

if "api_echo_sync" in config_data:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.vector_settings.api_echo_sync=config_data["api_echo_sync"]

def _apply_matrix_settings(self, matrix_data: Dict[str, Any]) -> None:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Apply matrix - specific settings"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""[BRAIN] Placeholder function - SHA - 256 ID=[autogen]"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""
if "reinforce_bad_vectors" in demo_data:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.reinforcement_settings.reinforce_bad_vectors=demo_data["reinforce_bad_vectors"]

if "log_ghost_trades" in demo_data:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.reinforcement_settings.log_ghost_trades=demo_data["log_ghost_trades"]

if "entropy_trigger_threshold" in demo_data:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
self.reinforcement_settings.entropy_trigger_threshold=demo_data["entropy_trigger_threshold"]

def _load_known_bad_vectors(self) -> List[Dict[str, Any]]:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Load known bad vectors map"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""
bad_vectors_path=self.config_path / "known_bad_vector_map.json"

if bad_vectors_path.exists():
        try:
        with open(bad_vectors_path, 'r') as f:
            pass  # Emergency placeholder
#                     return json.load(f)
        except Exception as e:
    pass  # TODO: Implement except block
safe_print("Warning: Could not load bad vectors map: {e}")

# Create default bad vectors map
default_bad_vectors = []
{}
"hash": "cafe23b4a1f8e9d2c5b7a3f6e9d2c5b7a3f6e9d2c5b7a3f6e9d2c5b7a3f6e9d2",
"tick_id": 12452,
"failure_type": "early_exit",
"matrix_id": "SFS8 - A5",
"timestamp": datetime.now().isoformat(),
        "confidence": 0.85
,
{}
"hash": "deadbeef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
"tick_id": 15678,
"failure_type": "false_positive",
"matrix_id": "SFS8 - A5",
"timestamp": datetime.now().isoformat(),
        "confidence": 0.92



with open(bad_vectors_path, 'w') as f:
        json.dump(default_bad_vectors, f, indent = 2)

#         return default_bad_vectors

def _initialize_matrix_weights(self) -> None:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Initialize matrix path weights for reinforcement learning"""Emergency consolidated docstring."""Emergency consolidated docstring."""
matrix_ids=["SFS8 - A5", "SFS16 - B3", "SFS42 - C7", "SFSS - D1", "SFSSS - E9"]

for matrix_id in matrix_ids:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Function implementation pending."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"logic_type": self.vector_settings.entry_logic,
"ghost_signal_weight": self.vector_settings.ghost_signal_weight,
"strict_mode": self.vector_settings.strict_mode_enabled,
"tick_delta_threshold": self.vector_settings.tick_delta_threshold,
"volume_sync": self.vector_settings.volume_sync_enabled,
"hash_confidence_min": self.vector_settings.hash_confidence_min,
"api_echo_sync": self.vector_settings.api_echo_sync


def get_exit_logic_config(self) -> Dict[str, Any]:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Get current exit logic configuration"""Emergency consolidated docstring."""Emergency consolidated docstring."""
#         return {}"""
"logic_type": self.vector_settings.exit_logic,
"ghost_signal_weight": self.vector_settings.ghost_signal_weight,
"strict_mode": self.vector_settings.strict_mode_enabled,
"tick_delta_threshold": self.vector_settings.tick_delta_threshold,
"volume_sync": self.vector_settings.volume_sync_enabled,
"hash_confidence_min": self.vector_settings.hash_confidence_min,
"api_echo_sync": self.vector_settings.api_echo_sync


def get_matrix_config(self) -> Dict[str, Any]:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Get current matrix configuration"""Emergency consolidated docstring."""Emergency consolidated docstring."""
#         return {}"""
"matrix_id": self.matrix_settings.matrix_id,
"entry_tolerance": self.matrix_settings.entry_tolerance,
"exit_flex": self.matrix_settings.exit_flex,
"priority_weight": self.matrix_settings.priority_weight,
"override_fault_controller": self.matrix_settings.override_fault_controller,
"bit_level": self.matrix_settings.bit_level,
"phase_count": self.matrix_settings.phase_count,
"thermal_limit": self.matrix_settings.thermal_limit,
"entropy_weight": self.matrix_settings.entropy_weight


def get_allocator_config(self) -> Dict[str, Any]:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Get current allocator configuration"""Emergency consolidated docstring."""Emergency consolidated docstring."""
#         return {}"""
"allocator_mode": self.allocator_settings.allocator_mode,
"long_weight": self.allocator_settings.long_weight,
"mid_weight": self.allocator_settings.mid_weight,
"short_weight": self.allocator_settings.short_weight,
"max_position_size": self.allocator_settings.max_position_size,
"correlation_limit": self.allocator_settings.correlation_limit,
"volatility_threshold": self.allocator_settings.volatility_threshold,
"auto_scaling_enabled": self.allocator_settings.auto_scaling_enabled


def get_reinforcement_config(self) -> Dict[str, Any]:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Get current reinforcement configuration"""Emergency consolidated docstring."""Emergency consolidated docstring."""
#         return {}"""
"enable_backlog_reinforcement": self.reinforcement_settings.enable_backlog_reinforcement,
"reinforce_bad_vectors": self.reinforcement_settings.reinforce_bad_vectors,
"log_ghost_trades": self.reinforcement_settings.log_ghost_trades,
"matrix_overlay": self.reinforcement_settings.matrix_overlay,
"entropy_trigger_threshold": self.reinforcement_settings.entropy_trigger_threshold,
"learning_rate": self.reinforcement_settings.learning_rate,
"memory_decay": self.reinforcement_settings.memory_decay,
"success_reward": self.reinforcement_settings.success_reward,
"failure_penalty": self.reinforcement_settings.failure_penalty


def get_fault_config(self) -> Dict[str, Any]:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Get current fault tolerance configuration"""Emergency consolidated docstring."""Emergency consolidated docstring."""
#         return {}"""
"fault_tolerance": self.fault_settings.fault_tolerance,
"enable_emergency_stop": self.fault_settings.enable_emergency_stop,
"max_drawdown_limit": self.fault_settings.max_drawdown_limit,
"thermal_management_enabled": self.fault_settings.thermal_management_enabled,
"performance_monitoring_enabled": self.fault_settings.performance_monitoring_enabled,
"debug_logging_enabled": self.fault_settings.debug_logging_enabled,
"experimental_mode": self.fault_settings.experimental_mode


def is_bad_vector(self, vector_hash: str, matrix_id: str) -> bool:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Check if a vector is in the known bad vectors map"""Emergency consolidated docstring."""Emergency consolidated docstring."""
for bad_vector in self.known_bad_vectors:"""
if (bad_vector["hash" == vector_hash and])
        bad_vector["matrix_id"] == matrix_id:
            pass  # Emergency placeholder
#                 return True
#         return False

def add_bad_vector(self, vector_hash: str, tick_id: int, failure_type: str,):
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""
bad_vector = {}"""
"hash": vector_hash,
"tick_id": tick_id,
"failure_type": failure_type,
"matrix_id": self.matrix_settings.matrix_id,
"timestamp": timestamp or datetime.now().isoformat(),
        "confidence": 0.8


self.known_bad_vectors.append(bad_vector)
        self._save_known_bad_vectors()

def _save_known_bad_vectors(self) -> None:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Save known bad vectors to file"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""
bad_vectors_path=self.config_path / "known_bad_vector_map.json"

with open(bad_vectors_path, 'w') as f:
        json.dump(self.known_bad_vectors, f, indent = 2)

def update_matrix_weights(self, matrix_id: str, success: bool) -> None:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Update matrix path weights based on success / failure"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Get current entropy trigger threshold"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""[BRAIN] Placeholder function - SHA - 256 ID=[autogen]"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""[BRAIN] Placeholder function - SHA - 256 ID = [autogen]"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""
"mode": "production",
"matrix_mode": self.matrix_settings.matrix_id,
"enable_backlog_reinforcement": self.reinforcement_settings.enable_backlog_reinforcement,
"fault_tolerance": self.fault_settings.fault_tolerance,
"api_echo_sync": self.vector_settings.api_echo_sync


with open(self.config_path / "main_settings.yaml", 'w') as f:
        yaml.dump(main_settings, f, default_flow_style = False)

# Save matrix settings
matrix_settings = {}
self.matrix_settings.matrix_id: {}
"entry_tolerance": self.matrix_settings.entry_tolerance,
"exit_flex": self.matrix_settings.exit_flex,
"priority_weight": self.matrix_settings.priority_weight,
"override_fault_controller": self.matrix_settings.override_fault_controller,
"bit_level": self.matrix_settings.bit_level,
"phase_count": self.matrix_settings.phase_count,
"thermal_limit": self.matrix_settings.thermal_limit,
"entropy_weight": self.matrix_settings.entropy_weight



with open(self.config_path / "matrix_settings.yaml", 'w') as f:
        yaml.dump(matrix_settings, f, default_flow_style = False)

# Save demo settings
demo_settings = {}
"mode": "demo",
"backtest_path": "./tests / demo_backlog/",
"reinforce_bad_vectors": self.reinforcement_settings.reinforce_bad_vectors,
"log_ghost_trades": self.reinforcement_settings.log_ghost_trades,
"matrix_overlay": self.reinforcement_settings.matrix_overlay,
"entropy_trigger_threshold": self.reinforcement_settings.entropy_trigger_threshold


with open(self.config_path / "demo_backtest_mode.yaml", 'w') as f:
        yaml.dump(demo_settings, f, default_flow_style = False)

safe_print("Settings saved successfully!")

except Exception as e:
    pass  # TODO: Implement except block
safe_print("Error saving settings: {e}")

def get_all_settings(self) -> Dict[str, Any]:
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
pass"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Get all current settings as a dictionary"""Emergency consolidated docstring."""Emergency consolidated docstring."""
#         return {}"""
"matrix_settings": asdict(self.matrix_settings),
        "vector_settings": asdict(self.vector_settings),
        "allocator_settings": asdict(self.allocator_settings),
        "reinforcement_settings": asdict(self.reinforcement_settings),
        "fault_settings": asdict(self.fault_settings),
        "matrix_path_weights": self.matrix_path_weights,
"known_bad_vectors_count": len(self.known_bad_vectors)



# Global settings controller instance
settings_controller = SettingsController()


def get_settings_controller() -> SettingsController:
        """
        """
            logger.error(f"Profit calculation failed: {e}")
            return 0.0
pass

"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""[BRAIN] Placeholder function - SHA - 256 ID=[autogen]"""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
"""Emergency consolidated docstring."""
"""
if __name__ == "__main__":
    """Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""Emergency placeholder docstring."""Emergency consolidated docstring."""Emergency consolidated docstring."""
safe_print("=== Schwabot Settings Controller Test ===")
    safe_print("Matrix ID: {controller.matrix_settings.matrix_id}")
    safe_print("Entry Logic: {controller.vector_settings.entry_logic}")
    safe_print("Allocator Mode: {controller.allocator_settings.allocator_mode}")
    safe_print("Reinforcement Enabled: {controller.reinforcement_settings.enable_backlog_reinforcement}")
    safe_print("Known Bad Vectors: {len(controller.known_bad_vectors)}")
    safe_print("Matrix Weights: {controller.matrix_path_weights}")

# Test bad vector detection
_test_hash = "cafe23b4a1f8e9d2c5b7a3f6e9d2c5b7a3f6e9d2c5b7a3f6e9d2c5b7a3f6e9d2"
is_bad=controller.is_bad_vector(test_hash, "SFS8 - A5")
    safe_print("Test hash is bad vector: {is_bad}")

# Test matrix weight update
controller.update_matrix_weights("SFS8 - A5", True)
    safe_print("Updated weight for SFS8 - A5: {controller.get_matrix_weight('SFS8 - A5')}")

safe_print("Settings controller test completed!")
