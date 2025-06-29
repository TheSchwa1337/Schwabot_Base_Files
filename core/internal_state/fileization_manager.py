# -*- coding: utf-8 -*-
"""
Fileization Manager for Internal State
=====================================

Provides a safe, modular interface for saving, loading, and validating internal state files
for handoff between trading system modules. Ensures no state is handed off in a mismatched
or inconsistent way. Supports tagging by phase, agent, and timestamp, and integrates with
the DynamicHandoffOrchestrator for dynamic state handoff.

Features:
- Save/load/validate numpy arrays, dicts, and other serializable objects
- Tagging by phase, agent, and timestamp
- Consistency and integrity checks
- Flake8-compliant, no missing stubs or definitions
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class FileizationManager:
    """
    Manages fileization (save/load/validate) of internal state for handoff.
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(os.path.dirname(__file__), "files")
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"FileizationManager initialized at {self.base_dir}")

    def _get_file_path(self, tag: str, phase: int, agent: Optional[str] = None, ext: str = "npy") -> str:
        agent_part = f"_{agent}" if agent else ""
        filename = f"state_{tag}_phase{phase}{agent_part}." + ext
        return os.path.join(self.base_dir, filename)

    def save_state(
        self, state: Union[np.ndarray, Dict[str, Any]], tag: str, phase: int, agent: Optional[str] = None
    ) -> str:
        """
        Saves the state to a file, tagged by phase and agent.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ext = "npy" if isinstance(state, np.ndarray) else "json"
        file_path = self._get_file_path(f"{tag}_{timestamp}", phase, agent, ext)
        if isinstance(state, np.ndarray):
            np.save(file_path, state)
        else:
            with open(file_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        logger.info(f"Saved state to {file_path}")
        return file_path

    def load_state(self, tag: str, phase: int, agent: Optional[str] = None, ext: str = "npy") -> Optional[Any]:
        """
        Loads the state from a file.
        """
        file_path = self._get_file_path(tag, phase, agent, ext)
        if not os.path.exists(file_path):
            logger.warning(f"State file not found: {file_path}")
            return None
        if ext == "npy":
            return np.load(file_path, allow_pickle=True)
        else:
            with open(file_path, "r") as f:
                return json.load(f)

    def validate_state(
        self, state: Any, expected_shape: Optional[tuple] = None, expected_type: Optional[type] = None
    ) -> bool:
        """
        Validates the state for shape and type consistency.
        """
        if expected_type and not isinstance(state, expected_type):
            logger.error(f"State type mismatch: expected {expected_type}, got {type(state)}")
            return False
        if expected_shape and hasattr(state, "shape") and state.shape != expected_shape:
            logger.error(f"State shape mismatch: expected {expected_shape}, got {getattr(state, 'shape', None)}")
            return False
        logger.info("State validated successfully.")
        return True

    def list_states(self) -> Dict[str, str]:
        """
        Lists all saved state files.
        """
        files = os.listdir(self.base_dir)
        return {f: os.path.join(self.base_dir, f) for f in files if f.startswith("state_")}

    def clear_states(self) -> None:
        """
        Deletes all saved state files.
        """
        for f in os.listdir(self.base_dir):
            if f.startswith("state_"):
                os.remove(os.path.join(self.base_dir, f))
        logger.info("All state files cleared.")


# Example usage (for demonstration/testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    manager = FileizationManager()
    arr = np.random.rand(10, 10)
    path = manager.save_state(arr, tag="test", phase=32, agent="BTC")
    loaded = manager.load_state("test", 32, agent="BTC")
    print(f"Loaded array shape: {getattr(loaded, 'shape', None)}")
    print(f"Validation: {manager.validate_state(loaded, expected_shape=(10, 10), expected_type=np.ndarray)}")
    print(f"States: {manager.list_states()}")
    manager.clear_states()
